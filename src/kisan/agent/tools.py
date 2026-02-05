"""Tool definitions for the Kisan agent."""

from typing import Any

from kisan.core.config import Settings, get_settings
from kisan.core.exceptions import MandiAPIError, RetrievalError, ToolExecutionError, VisionError
from kisan.core.logging import logger
from kisan.modules.disease.detector import DiseaseDetector
from kisan.modules.mandi.client import MandiClient
from kisan.modules.schemes.retriever import SchemeRetriever
from kisan.services.llm import LLMService
from kisan.services.vectordb import VectorDBService

# Tool definitions for OpenAI function calling
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "detect_disease",
            "description": (
                "Analyze a crop image to detect diseases and provide treatment. "
                "Use this when the user provides an image of their crop."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "crop_hint": {
                        "type": "string",
                        "description": "Optional hint about what crop is in the image",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_mandi_prices",
            "description": (
                "Get current market prices for agricultural commodities from Indian mandis. "
                "Use when the user asks about crop prices."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "commodity": {
                        "type": "string",
                        "description": "Crop/commodity to get prices for (wheat, rice, onion)",
                    },
                    "state": {
                        "type": "string",
                        "description": "State to filter by (Delhi, Maharashtra)",
                    },
                    "district": {
                        "type": "string",
                        "description": "District to filter prices by",
                    },
                },
                "required": ["commodity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_schemes",
            "description": (
                "Search for government agricultural schemes and subsidies. "
                "Use for PM-KISAN, crop insurance, or government program questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The question about government schemes",
                    }
                },
                "required": ["query"],
            },
        },
    },
]


class ToolExecutor:
    """Executes agent tools."""

    def __init__(
        self,
        llm_service: LLMService,
        vectordb_service: VectorDBService,
        settings: Settings | None = None,
    ):
        self.settings = settings or get_settings()
        self.llm = llm_service
        self.vectordb = vectordb_service

        # Initialize tool handlers
        self.disease_detector = DiseaseDetector(llm_service, self.settings)
        self.mandi_client = MandiClient(self.settings)
        self.scheme_retriever = SchemeRetriever(
            llm_service, vectordb_service, self.settings
        )

        self._current_image: str | None = None

    def set_current_image(self, image_base64: str | None) -> None:
        """Set the current image for disease detection."""
        self._current_image = image_base64

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:
        """Execute a tool and return the result as a string."""
        logger.info(f"Executing tool: {tool_name} with args: {arguments}")

        try:
            if tool_name == "detect_disease":
                return await self._execute_disease_detection(arguments)
            elif tool_name == "get_mandi_prices":
                return await self._execute_mandi_prices(arguments)
            elif tool_name == "search_schemes":
                return await self._execute_scheme_search(arguments)
            else:
                raise ToolExecutionError(tool_name, f"Unknown tool: {tool_name}")

        except ToolExecutionError:
            raise
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            raise ToolExecutionError(tool_name, str(e)) from e

    async def _execute_disease_detection(self, arguments: dict) -> str:
        """Execute disease detection tool."""
        if not self._current_image:
            return (
                "No image was provided. Please upload an image of your crop "
                "so I can analyze it for diseases."
            )

        try:
            crop_hint = arguments.get("crop_hint")
            result = await self.disease_detector.detect(
                image_base64=self._current_image,
                crop_hint=crop_hint,
            )

            # Format result for LLM
            output = []
            if result.crop_identified:
                output.append(f"**Crop Identified:** {result.crop_identified}")

            if result.disease_detected:
                output.append("**Disease Detected:** Yes")
                if result.disease_name:
                    output.append(f"**Disease Name:** {result.disease_name}")
                output.append(f"**Confidence:** {result.confidence:.0%}")
                output.append(f"**Severity:** {result.severity.value}")

                if result.symptoms:
                    output.append(f"**Symptoms:** {', '.join(result.symptoms[:5])}")
                if result.causes:
                    output.append(f"**Causes:** {', '.join(result.causes[:3])}")
                if result.treatments:
                    treatments_text = "; ".join(
                        t.method for t in result.treatments[:3]
                    )
                    output.append(f"**Treatments:** {treatments_text}")
                if result.prevention_tips:
                    output.append(
                        f"**Prevention:** {', '.join(result.prevention_tips[:3])}"
                    )
            else:
                output.append("**Health Status:** Healthy - No disease detected")

            if result.additional_notes:
                output.append(f"**Notes:** {result.additional_notes[:200]}")

            return "\n".join(output)

        except VisionError as e:
            return f"Could not analyze the image: {e.message}"

    async def _execute_mandi_prices(self, arguments: dict) -> str:
        """Execute mandi price lookup tool."""
        commodity = arguments.get("commodity", "")
        if not commodity:
            return "Please specify which commodity you want prices for."

        try:
            result = await self.mandi_client.get_prices(
                commodity=commodity,
                state=arguments.get("state"),
                district=arguments.get("district"),
            )

            if not result.prices:
                return (
                    f"No price data found for {commodity}. "
                    "This could be because:\n"
                    "- The commodity name might be spelled differently\n"
                    "- Price data may not be available for this commodity\n"
                    "- Try specifying a different state or district"
                )

            # Format prices for display
            output = [f"**Market Prices for {commodity.title()}**\n"]

            for price in result.prices[:10]:  # Limit to 10 results
                output.append(
                    f"📍 **{price.market}** ({price.district}, {price.state})\n"
                    f"   Price Range: ₹{price.min_price:,.0f} - ₹{price.max_price:,.0f}/quintal\n"
                    f"   Modal Price: ₹{price.modal_price:,.0f}/quintal"
                )

            if result.total_results > 10:
                output.append(f"\n*Showing 10 of {result.total_results} results*")

            return "\n".join(output)

        except MandiAPIError as e:
            return f"Could not fetch mandi prices: {e.message}"

    async def _execute_scheme_search(self, arguments: dict) -> str:
        """Execute scheme search tool."""
        query = arguments.get("query", "")
        if not query:
            return "Please specify what scheme information you're looking for."

        try:
            result = await self.scheme_retriever.query(query)

            if not result.answer:
                return (
                    "I couldn't find specific information about this. "
                    "Try asking about specific schemes like PM-KISAN, PMFBY, "
                    "Kisan Credit Card, or crop insurance."
                )

            output = [result.answer]

            if result.schemes_mentioned:
                output.append(f"\n**Schemes mentioned:** {', '.join(result.schemes_mentioned)}")

            if result.documents:
                sources = list(set(d.source for d in result.documents[:3]))
                output.append(f"\n*Sources: {', '.join(sources)}*")

            return "\n".join(output)

        except RetrievalError as e:
            return f"Could not search schemes: {e.message}"
