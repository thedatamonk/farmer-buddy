"""Parser for mandi API responses."""

from datetime import datetime

from kisan.core.logging import logger
from kisan.schemas.mandi import MandiPrice


def parse_mandi_response(data: dict) -> list[MandiPrice]:
    """Parse raw API response into MandiPrice objects."""
    prices = []

    records = data.get("records", [])
    if not records:
        logger.debug("No records found in mandi API response")
        return prices

    for record in records:
        try:
            arrival_date = None
            date_str = record.get("arrival_date", "")
            if date_str:
                try:
                    arrival_date = datetime.strptime(date_str, "%d/%m/%Y").date()
                except ValueError:
                    try:
                        arrival_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                    except ValueError:
                        pass

            price = MandiPrice(
                commodity=record.get("commodity", "Unknown"),
                variety=record.get("variety"),
                state=record.get("state", "Unknown"),
                district=record.get("district", "Unknown"),
                market=record.get("market", "Unknown"),
                min_price=float(record.get("min_price", 0)),
                max_price=float(record.get("max_price", 0)),
                modal_price=float(record.get("modal_price", 0)),
                arrival_date=arrival_date,
            )
            prices.append(price)

        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse mandi record: {e}")
            continue

    return prices


def format_price_for_display(price: MandiPrice) -> str:
    """Format a single price record for display."""
    lines = [
        f"📍 {price.market}, {price.district}, {price.state}",
        f"🌾 {price.commodity}" + (f" ({price.variety})" if price.variety else ""),
        "💰 Prices (₹/quintal):",
        f"   Min: ₹{price.min_price:,.0f}",
        f"   Max: ₹{price.max_price:,.0f}",
        f"   Modal: ₹{price.modal_price:,.0f}",
    ]
    if price.arrival_date:
        lines.append(f"📅 Date: {price.arrival_date.strftime('%d %b %Y')}")
    return "\n".join(lines)


def format_prices_summary(prices: list[MandiPrice], commodity: str) -> str:
    """Format multiple prices into a summary."""
    if not prices:
        return f"No price data found for {commodity}."

    # Group by state
    by_state: dict[str, list[MandiPrice]] = {}
    for p in prices:
        if p.state not in by_state:
            by_state[p.state] = []
        by_state[p.state].append(p)

    lines = [f"## Market Prices for {commodity.title()}\n"]

    for state, state_prices in sorted(by_state.items()):
        lines.append(f"\n### {state}")
        for p in state_prices[:3]:  # Max 3 per state
            lines.append(
                f"- **{p.market}** ({p.district}): "
                f"₹{p.min_price:,.0f} - ₹{p.max_price:,.0f} "
                f"(Modal: ₹{p.modal_price:,.0f})"
            )

    # Calculate overall stats
    all_modal = [p.modal_price for p in prices if p.modal_price > 0]
    if all_modal:
        avg_price = sum(all_modal) / len(all_modal)
        min_price = min(all_modal)
        max_price = max(all_modal)
        lines.append("\n### Summary")
        lines.append(f"- Average modal price: ₹{avg_price:,.0f}/quintal")
        lines.append(f"- Price range: ₹{min_price:,.0f} - ₹{max_price:,.0f}")
        lines.append(f"- Markets covered: {len(prices)}")

    return "\n".join(lines)
