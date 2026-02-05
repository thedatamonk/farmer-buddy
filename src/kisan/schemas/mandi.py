"""Mandi (market) price related Pydantic schemas."""

from datetime import date

from pydantic import BaseModel, Field


class MandiPrice(BaseModel):
    """Price information for a commodity at a mandi."""

    commodity: str = Field(..., description="Name of the commodity")
    variety: str | None = Field(default=None, description="Variety of the commodity")
    state: str = Field(..., description="State name")
    district: str = Field(..., description="District name")
    market: str = Field(..., description="Market/Mandi name")
    min_price: float = Field(..., description="Minimum price in INR per quintal")
    max_price: float = Field(..., description="Maximum price in INR per quintal")
    modal_price: float = Field(..., description="Modal (most common) price in INR per quintal")
    arrival_date: date | None = Field(default=None, description="Date of price recording")


class MandiPriceResult(BaseModel):
    """Result of mandi price query."""

    query_commodity: str = Field(..., description="Commodity that was queried")
    query_location: str | None = Field(default=None, description="Location that was queried")
    prices: list[MandiPrice] = Field(default_factory=list, description="List of prices found")
    total_results: int = Field(default=0, description="Total number of results")
    message: str | None = Field(default=None, description="Additional message or status")


class MandiPriceRequest(BaseModel):
    """Request for mandi price lookup."""

    commodity: str = Field(..., min_length=1, description="Commodity to search for")
    state: str | None = Field(default=None, description="State to filter by")
    district: str | None = Field(default=None, description="District to filter by")
    market: str | None = Field(default=None, description="Specific market to search")
