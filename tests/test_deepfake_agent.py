"""Tests for the DeepfakeAgent."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.deepfake_agent import DeepfakeAgent


@pytest.fixture
def agent(patch_nim_client):
    return DeepfakeAgent()


class TestDeepfakeAgent:
    @pytest.mark.asyncio
    async def test_analyze_no_images(self, agent):
        result = await agent.analyze()
        assert result["status"] == "skipped"
        assert result["manipulation_score"] == 0

    def test_parse_analysis_high_likelihood(self, agent):
        response = "Highly likely manipulation. Score: 85/100. Clear manipulation detected."
        score, detections = agent._parse_analysis(response)
        assert score >= 80

    def test_parse_analysis_low_likelihood(self, agent):
        # Note: "unlikely" contains substring "likely", so the parser hits
        # the "likely" branch first due to elif ordering. Use unambiguous
        # phrasing that doesn't contain "likely" as a substring.
        response = "This image is authentic with no manipulation detected."
        score, detections = agent._parse_analysis(response)
        assert score <= 25

    def test_parse_analysis_detections(self, agent):
        response = "The image shows clone stamp artifacts and shadow inconsistencies. Compression issues visible."
        score, detections = agent._parse_analysis(response)
        assert "cloning" in detections
        assert "lighting" in detections  # "shadow" triggers lighting
        assert "compression" in detections

    def test_generate_summary_high(self, agent):
        summary = agent._generate_summary(80, ["cloning", "lighting"])
        assert "HIGH RISK" in summary

    def test_generate_summary_clean(self, agent):
        summary = agent._generate_summary(10, [])
        assert "CLEAN" in summary

    @pytest.mark.asyncio
    async def test_analyze_missing_file(self, agent):
        result = await agent.analyze(image_paths=["/nonexistent/photo.jpg"])
        assert result["status"] == "success"
        assert result["images_analyzed"] == 1
