import unittest
from unittest.mock import patch

import web_evidence
import web_search


WHO_STROKE_URL = "https://www.who.int/news-room/fact-sheets/detail/stroke"


def result(title, url, snippet=""):
    return web_search.SearchResult(title=title, url=url, snippet=snippet)


class WebEvidenceTests(unittest.TestCase):
    def test_who_query_is_planned_as_site_constrained_english_query(self):
        plan = web_evidence.plan_web_search(
            "查询 WHO 关于卒中的公开资料并提供来源"
        )

        self.assertEqual(plan.required_domains, ("who.int",))
        self.assertEqual(plan.queries, ("site:who.int stroke fact sheet",))
        self.assertTrue(plan.source_lookup)
        self.assertNotIn("whois", plan.queries[0].lower())

    def test_whois_does_not_trigger_who_organization_constraint(self):
        plan = web_evidence.plan_web_search("查询 WHOIS 域名注册信息")

        self.assertEqual(plan.required_domains, ())

    def test_patient_case_does_not_use_source_lookup_fast_route(self):
        self.assertFalse(
            web_evidence.is_source_lookup_query(
                "患者突发偏瘫，请结合卒中指南分析诊断与治疗"
            )
        )

    @patch.object(web_evidence, "search_web")
    def test_filters_whois_noise_and_keeps_official_topic_result(self, search):
        search.return_value = web_search.SearchResponse(
            provider="brave",
            results=(
                result(
                    "WHOIS Lookup",
                    "https://www.godaddy.com/whois",
                    "Find domain registration data.",
                ),
                result(
                    "Stroke",
                    WHO_STROKE_URL,
                    "Stroke is a leading cause of death and disability.",
                ),
                result(
                    "Unrelated WHO page",
                    "https://www.who.int/about",
                    "About the World Health Organization.",
                ),
            ),
        )

        response = web_evidence.search_web_for_question(
            "查询 WHO 关于卒中的公开资料并提供来源"
        )

        search.assert_called_once_with("site:who.int stroke fact sheet")
        self.assertEqual([item.url for item in response.results], [WHO_STROKE_URL])
        self.assertEqual(response.rejected_count, 2)

    @patch.object(web_evidence, "search_web")
    def test_non_relevant_provider_response_is_reported_as_failure(self, search):
        search.return_value = web_search.SearchResponse(
            provider="brave",
            results=(
                result("WHOIS", "https://whois.example/lookup", "domain"),
            ),
        )

        with self.assertRaisesRegex(
            web_search.WebSearchError,
            "没有返回与当前问题匹配的可信证据",
        ):
            web_evidence.search_web_for_question(
                "查询 WHO 关于卒中的公开资料并提供来源"
            )

    def test_direct_answer_exposes_query_ids_and_exact_urls(self):
        response = web_evidence.EvidenceSearchResponse(
            provider="brave",
            results=[
                result(
                    "Stroke",
                    WHO_STROKE_URL,
                    "WHO fact sheet summary.",
                )
            ],
            queries=("site:who.int stroke fact sheet",),
            required_domains=("who.int",),
        )

        summary = web_evidence.format_evidence_summary(response)
        answer = web_evidence.build_source_lookup_answer("WHO 卒中资料来源", response)

        self.assertIn("实际检索式：site:who.int stroke fact sheet", summary)
        self.assertIn("[W1]", summary)
        self.assertIn("[W1]", answer)
        self.assertIn(WHO_STROKE_URL, answer)
        self.assertNotIn("未找到", answer)

    def test_parentheses_in_approved_url_are_preserved(self):
        cvd_url = (
            "https://www.who.int/en/news-room/fact-sheets/detail/"
            "cardiovascular-diseases-(CVDs)"
        )
        response = web_evidence.EvidenceSearchResponse(
            provider="brave",
            results=[result("Cardiovascular diseases (CVDs)", cvd_url)],
            queries=("site:who.int stroke fact sheet",),
        )
        answer = f"- [W1] [Cardiovascular diseases (CVDs)]({cvd_url})"

        cleaned = web_evidence.canonicalize_grounded_answer(answer, response)

        self.assertEqual(cleaned, answer)
        self.assertEqual(web_evidence.grounding_issues(cleaned, response), [])
        self.assertNotIn("未验证链接已移除", cleaned)

    def test_grounding_canonicalizer_removes_invented_url(self):
        response = web_evidence.EvidenceSearchResponse(
            provider="brave",
            results=[result("Stroke", WHO_STROKE_URL)],
            queries=("site:who.int stroke",),
        )
        unsafe = (
            "请参考 [猜测页面](https://www.who.int/fake-guideline)，"
            "但当前未找到相关资料。"
        )

        cleaned = web_evidence.canonicalize_grounded_answer(unsafe, response)
        issues = web_evidence.grounding_issues(cleaned, response)

        self.assertNotIn("https://www.who.int/fake-guideline", cleaned)
        self.assertIn(WHO_STROKE_URL, cleaned)
        self.assertEqual(issues, [])
        self.assertNotIn("未找到相关资料", cleaned)


if __name__ == "__main__":
    unittest.main()
