"""Focused tests for source-backed research validation and orchestration."""

import asyncio
import json
import os
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

os.environ.setdefault("OPENAI_API_KEY", "test-key")

import home
from test_fact_opinion_route import make_result


def research_claim(
    *,
    verdict: str = "supported",
    url: str = "https://example.gov/report",
    source_type: str = "official",
) -> dict:
    return {
        "claim": "The published rate was four percent.",
        "verdict": verdict,
        "evidence_summary": "The official report lists the published rate as four percent.",
        "sources": [
            {
                "title": "Official rate report",
                "url": url,
                "source_type": source_type,
                "relevance_summary": (
                    "The report directly publishes the rate discussed in the claim."
                ),
            }
        ],
    }


def research_output(*claims: dict) -> dict:
    return {
        "claims": list(claims),
        "overall_reliability": "high",
        "notes": "Only the returned high-priority factual claims were checked.",
    }


def web_response(parsed: dict, *urls: str, annotation_urls: tuple[str, ...] = ()) -> dict:
    output = [
        {
            "type": "web_search_call",
            "status": "completed",
            "action": {
                "type": "search",
                "sources": [{"type": "url", "url": url} for url in urls],
            },
        }
    ]
    if annotation_urls:
        output.append(
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": json.dumps(parsed),
                        "annotations": [
                            {"type": "url_citation", "url": url}
                            for url in annotation_urls
                        ],
                    }
                ],
            }
        )
    return {
        "output_text": json.dumps(parsed),
        "output": output,
    }


class StrictResearchValidationTests(unittest.TestCase):
    def test_decisive_verdict_requires_a_source(self):
        raw = research_output(research_claim())
        raw["claims"][0]["sources"] = []

        with self.assertRaises(HTTPException) as raised:
            home.validate_ai_research(raw, candidate_claim_count=1)

        self.assertEqual(raised.exception.status_code, 502)

    def test_source_requires_http_or_https(self):
        raw = research_output(research_claim(url="ftp://example.gov/report"))

        with self.assertRaises(HTTPException):
            home.validate_ai_research(raw, candidate_claim_count=1)

    def test_decisive_verdict_rejects_only_other_quality_sources(self):
        raw = research_output(research_claim(source_type="other"))

        with self.assertRaises(HTTPException):
            home.validate_ai_research(raw, candidate_claim_count=1)

    def test_coverage_discloses_checked_unchecked_and_truncation(self):
        result = home.validate_ai_research(
            research_output(research_claim()),
            candidate_claim_count=4,
            total_factual_characters=home.MAX_RESEARCH_INPUT_CHARS + 200,
        )

        self.assertEqual(result.coverage.status, "partial")
        self.assertEqual(result.coverage.checked_claim_count, 1)
        self.assertEqual(result.coverage.unchecked_claim_count, 3)
        self.assertTrue(result.coverage.input_truncated)
        self.assertIn("does not imply", result.coverage.scope_note)

    def test_no_facts_are_explicitly_not_assessed(self):
        result = home.no_factual_research_result()

        self.assertEqual(result.overall_reliability, "not_assessed")
        self.assertEqual(result.coverage.status, "none")
        self.assertEqual(result.coverage.checked_claim_count, 0)

    def test_no_facts_disclose_an_unclassified_article_tail(self):
        result = home.no_factual_research_result(article_input_truncated=True)

        self.assertTrue(result.coverage.input_truncated)
        self.assertIn("outside that window", result.coverage.scope_note)


class ResearchToolOrchestrationTests(unittest.TestCase):
    def test_research_forces_web_search_and_passes_bias_output(self):
        parsed = research_output(research_claim())
        response = web_response(parsed, "https://example.gov/report")
        bias = home.no_factual_bias_result()

        with patch.object(
            home, "run_model_json", new=AsyncMock(return_value=response)
        ) as run:
            result = asyncio.run(
                home.researcher_ai(
                    'A source said "The rate was four percent."',
                    title="Rates",
                    bias_result=bias,
                    candidate_claim_count=1,
                )
            )

        self.assertEqual(result, parsed)
        call = run.await_args.kwargs
        self.assertEqual(call["tools"], [{"type": "web_search"}])
        self.assertEqual(call["tool_choice"], {"type": "web_search"})
        self.assertEqual(call["include"], ["web_search_call.action.sources"])
        self.assertEqual(call["model"], "gpt-6.1-terra")
        self.assertEqual(call["reasoning"], {"effort": "low"})
        self.assertEqual(
            call["payload"]["bias_detector_output"], bias.model_dump(mode="json")
        )
        self.assertEqual(
            call["payload"]["quoted_spans"][0]["text"],
            "The rate was four percent.",
        )

    def test_research_rejects_original_article_and_common_aliases(self):
        original = "https://www.cnn.com/2026/11/15/politics/story/index.html"
        aliases = [original, original + "?utm_source=search#top",
                   original + "?ref=homepage&cid=external",
                   "http://cnn.com/2026/11/15/politics/story/",
                   "https://edition.cnn.com/2026/11/15/politics/story/index.html",
                   "https://amp.cnn.com/cnn/2026/11/15/politics/story/index.html",
                   "https://www.cnn.com/2026/11/15/politics/story/amp?output=amp"]
        for url in aliases:
            with self.subTest(url=url):
                parsed = research_output(research_claim(url=url, source_type="reputable_secondary"))
                response = web_response(parsed, url)
                with patch.object(home, "run_model_json", new=AsyncMock(return_value=response)) as run:
                    result = asyncio.run(home.researcher_ai("A factual statement.", source_url=original))
                self.assertEqual(run.await_args.kwargs["payload"]["source_url"], original)
                self.assertEqual(result["claims"][0]["sources"], [])
                self.assertEqual(result["claims"][0]["verdict"], "unclear")
                self.assertIn("self-citations", result["notes"])

    def test_independent_evidence_survives_original_article_removal(self):
        original = "https://www.cnn.com/story"
        claim = research_claim(url=original, source_type="reputable_secondary")
        claim["sources"].extend(research_claim(url="https://example.gov/report")["sources"])
        parsed = research_output(claim)
        urls = {home._normalize_provenance_url(original), "https://example.gov/report"}
        result, dropped, downgraded = home._sanitize_research_url_provenance(parsed, urls, original)
        self.assertEqual(dropped, 1);self.assertEqual(downgraded, 0)
        self.assertEqual(result["claims"][0]["verdict"], "supported")
        self.assertEqual([s["url"] for s in result["claims"][0]["sources"]], ["https://example.gov/report"])

    def test_different_article_and_distinct_query_documents_are_not_self_citations(self):
        self.assertFalse(home._is_original_research_source("https://cnn.com/story-y", "https://www.cnn.com/story-x"))
        self.assertFalse(home._is_original_research_source("https://example.gov/report?id=2", "https://example.gov/report?id=1"))
        self.assertFalse(home._is_original_research_source("https://example.gov/report", ""))
        self.assertIsNone(home._normalize_provenance_url("https://example.gov:invalid/report"))

    def test_distinct_source_articles_do_not_share_a_legacy_job_key(self):
        first = home.ArticleJobRequest(page_url="https://cnn.com/story-x", text="Same text", title="Title")
        second = home.ArticleJobRequest(page_url="https://cnn.com/story-y", text="Same text", title="Title")
        self.assertNotEqual(home._article_content_key(first), home._article_content_key(second))

    def test_research_rejects_response_without_completed_web_search(self):
        response = {
            "output_text": json.dumps(research_output(research_claim())),
            "output": [{"type": "message", "content": []}],
        }

        with patch.object(
            home, "run_model_json", new=AsyncMock(return_value=response)
        ):
            result = asyncio.run(home.researcher_ai("A factual statement."))

        self.assertEqual(result["error_code"], "research_no_web_search")
        self.assertNotIn("completed", result["error"])

    def test_research_removes_unverified_citation_and_downgrades_claim(self):
        parsed = research_output(research_claim(url="https://invented.example/report"))
        response = web_response(parsed, "https://example.gov/report")

        with patch.object(
            home, "run_model_json", new=AsyncMock(return_value=response)
        ):
            result = asyncio.run(home.researcher_ai("A factual statement."))

        self.assertEqual(result["claims"][0]["verdict"], "unclear")
        self.assertEqual(result["claims"][0]["sources"], [])
        self.assertEqual(result["overall_reliability"], "low")
        self.assertIn("Unverified citation links were removed", result["notes"])

    def test_research_accepts_url_citation_annotations_without_action_sources(self):
        parsed = research_output(research_claim(url="https://EXAMPLE.gov/report/"))
        response = web_response(
            parsed, annotation_urls=("https://example.gov/report",)
        )

        with patch.object(
            home, "run_model_json", new=AsyncMock(return_value=response)
        ):
            result = asyncio.run(home.researcher_ai("A factual statement."))

        self.assertEqual(result, parsed)

    def test_research_accepts_completed_action_url(self):
        for action_type in ("open_page", "find_in_page"):
            with self.subTest(action_type=action_type):
                parsed = research_output(research_claim())
                response = web_response(parsed, "https://search.example/result")
                response["output"].insert(
                    1,
                    {
                        "type": "web_search_call",
                        "status": "completed",
                        "action": {
                            "type": action_type,
                            "url": "https://example.gov/report",
                        },
                    },
                )

                with patch.object(
                    home, "run_model_json", new=AsyncMock(return_value=response)
                ):
                    result = asyncio.run(
                        home.researcher_ai("A factual statement.")
                    )

                self.assertEqual(result, parsed)

    def test_research_accepts_equivalent_url_with_tracking_parameters(self):
        parsed = research_output(
            research_claim(url="https://example.gov/report?section=2")
        )
        response = web_response(
            parsed,
            "https://example.gov/report?utm_source=search&section=2&gclid=abc",
        )

        with patch.object(
            home, "run_model_json", new=AsyncMock(return_value=response)
        ):
            result = asyncio.run(home.researcher_ai("A factual statement."))

        self.assertEqual(result, parsed)

    def test_research_does_not_trust_incomplete_action_url(self):
        parsed = research_output(research_claim())
        response = web_response(parsed, "https://search.example/result")
        response["output"].insert(
            1,
            {
                "type": "web_search_call",
                "status": "incomplete",
                "action": {
                    "type": "find_in_page",
                    "url": "https://example.gov/report",
                },
            },
        )

        with patch.object(
            home, "run_model_json", new=AsyncMock(return_value=response)
        ):
            result = asyncio.run(home.researcher_ai("A factual statement."))

        self.assertEqual(result["claims"][0]["verdict"], "unclear")
        self.assertEqual(result["claims"][0]["sources"], [])

    def test_research_completed_search_without_citations_returns_safe_unclear_claim(self):
        parsed = research_output(research_claim())
        response = web_response(parsed)

        with patch.object(
            home, "run_model_json", new=AsyncMock(return_value=response)
        ):
            result = asyncio.run(home.researcher_ai("A factual statement."))

        self.assertEqual(result["claims"][0]["verdict"], "unclear")
        self.assertEqual(result["claims"][0]["sources"], [])
        self.assertEqual(result["overall_reliability"], "low")

    def test_research_retries_token_truncation_with_remaining_global_budget(self):
        parsed = research_output(research_claim())
        incomplete = {
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [],
        }
        complete = web_response(parsed, "https://example.gov/report")

        with patch.object(
            home,
            "run_model_json",
            new=AsyncMock(side_effect=[incomplete, complete]),
        ) as run:
            result = asyncio.run(
                home.researcher_ai(
                    "A factual statement.", timeout_seconds=60.0
                )
            )

        self.assertEqual(result, parsed)
        self.assertEqual(run.await_count, 2)
        self.assertEqual(
            [call.kwargs["max_tokens"] for call in run.await_args_list],
            [1800, 2600],
        )
        self.assertGreater(run.await_args_list[0].kwargs["timeout"], 30.0)
        self.assertGreater(run.await_args_list[1].kwargs["timeout"], 30.0)

    def test_analyze_research_does_not_require_bias_output(self):
        text = ("The rate was four percent. " * 10).strip()
        classification = make_result(
            text, [("fact", []) for _ in home.segment_article(text)]
        )
        bias_output = home.no_factual_bias_result().model_dump(mode="json")
        raw_research = research_output(research_claim())
        request = home.AnalyzeRequest(text=text, title="Rates", source_url="https://cnn.com/source-story")

        with (
            patch.object(
                home,
                "classify_article_fact_opinion",
                new=AsyncMock(return_value=classification),
            ),
            patch.object(
                home, "analyze_bias", new=AsyncMock(return_value=bias_output)
            ) as bias,
            patch.object(
                home, "researcher_ai", new=AsyncMock(return_value=raw_research)
            ) as research,
        ):
            response = asyncio.run(home.analyze(request))

        bias.assert_awaited_once()
        self.assertEqual(research.await_args.kwargs["source_url"], "https://cnn.com/source-story")
        self.assertNotIn("bias_result", research.await_args.kwargs)
        self.assertEqual(response["ai_research"].coverage.status, "partial")

    def test_article_paths_start_bias_and_research_concurrently(self):
        text = ("The published rate was four percent. " * 8).strip()
        classification = make_result(text, [("fact", []) for _ in home.segment_article(text)])
        for mode in ["analyze", "job"]:
            with self.subTest(mode=mode):
                async def scenario():
                    bias_started, research_started = asyncio.Event(), asyncio.Event()
                    async def bias(*args, **kwargs):
                        bias_started.set()
                        await asyncio.wait_for(research_started.wait(), 1)
                        return home.no_factual_bias_result().model_dump(mode="json")
                    async def research(*args, **kwargs):
                        research_started.set()
                        await asyncio.wait_for(bias_started.wait(), 1)
                        self.assertNotIn("bias_result", kwargs)
                        return research_output(research_claim())
                    with (
                        patch.object(home, "classify_article_fact_opinion", new=AsyncMock(return_value=classification)),
                        patch.object(home, "analyze_bias", side_effect=bias),
                        patch.object(home, "researcher_ai", side_effect=research),
                    ):
                        if mode == "analyze":
                            await home.analyze(home.AnalyzeRequest(text=text))
                        else:
                            key = "parallel-research-test"
                            home.article_jobs[key] = {"status": "queued"}
                            try:
                                await home.run_article_job(key, home.ArticleJobRequest(page_url="https://example.com/story", text=text))
                                self.assertEqual(home.article_jobs[key]["status"], "complete")
                            finally:
                                home.article_jobs.pop(key, None)
                    self.assertTrue(bias_started.is_set() and research_started.is_set())
                asyncio.run(scenario())

    def test_bias_failure_cancels_owned_research_in_both_article_paths(self):
        text = ("The published rate was four percent. " * 8).strip()
        classification = make_result(text, [("fact", []) for _ in home.segment_article(text)])
        for mode in ["analyze", "job"]:
            with self.subTest(mode=mode):
                async def scenario():
                    started, stopped = asyncio.Event(), asyncio.Event()
                    async def research(*args, **kwargs):
                        started.set()
                        try:
                            await asyncio.Event().wait()
                        finally:
                            stopped.set()
                    async def bias(*args, **kwargs):
                        await asyncio.wait_for(started.wait(), 1)
                        return {"error": "Fixture bias failure."}
                    with (
                        patch.object(home, "classify_article_fact_opinion", new=AsyncMock(return_value=classification)),
                        patch.object(home, "analyze_bias", side_effect=bias),
                        patch.object(home, "researcher_ai", side_effect=research),
                    ):
                        if mode == "analyze":
                            with self.assertRaises(HTTPException):
                                await home.analyze(home.AnalyzeRequest(text=text))
                        else:
                            key = "cancel-research-test"
                            home.article_jobs[key] = {"status": "queued"}
                            try:
                                await home.run_article_job(key, home.ArticleJobRequest(page_url="https://example.com/story", text=text))
                                self.assertEqual(home.article_jobs[key]["status"], "failed")
                            finally:
                                home.article_jobs.pop(key, None)
                    self.assertTrue(stopped.is_set(), "Research must not outlive a failed analysis")
                asyncio.run(scenario())

    def test_article_job_completes_when_research_is_unavailable(self):
        text = ("The published rate was four percent. " * 8).strip()
        classification = make_result(
            text, [("fact", []) for _ in home.segment_article(text)]
        )
        job_id = "research-partial-test"
        home.article_jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "result": None,
            "error": None,
        }
        request = home.ArticleJobRequest(
            page_url="https://example.gov/article", text=text, title="Rates"
        )

        try:
            with (
                patch.object(
                    home,
                    "classify_article_fact_opinion",
                    new=AsyncMock(return_value=classification),
                ),
                patch.object(
                    home,
                    "analyze_bias",
                    new=AsyncMock(
                        return_value=home.no_factual_bias_result().model_dump(mode="json")
                    ),
                ),
                patch.object(
                    home,
                    "researcher_ai",
                    new=AsyncMock(
                        return_value={
                            "error": "Research verification failed. Please try again.",
                            "error_code": "research_unverified_citation",
                            "error_id": "safe-reference",
                        }
                    ),
                ) as research_call,
            ):
                asyncio.run(home.run_article_job(job_id, request))

            self.assertEqual(research_call.await_args.kwargs["source_url"], "https://example.gov/article")
            job = home.article_jobs[job_id]
            self.assertEqual(job["status"], "complete")
            self.assertIsNone(job["error"])
            self.assertIsNotNone(job["result"]["ai_result"])
            self.assertEqual(job["result"]["status"], "analyzed")
            research = job["result"]["ai_research"]
            self.assertEqual(research["claims"], [])
            self.assertEqual(research["overall_reliability"], "not_assessed")
            self.assertEqual(research["coverage"]["checked_claim_count"], 0)
            self.assertGreater(research["coverage"]["candidate_claim_count"], 0)
            self.assertIn("verification was unavailable", research["notes"])
        finally:
            home.article_jobs.pop(job_id, None)


if __name__ == "__main__":
    unittest.main()
