import logging

from loguru import logger
from mcp.server.fastmcp import FastMCP

logger.remove()
logging.getLogger().setLevel(logging.CRITICAL)

mcp = FastMCP("slow_clap_demo")


@mcp.tool()
def slow_clap(claps: int = 1) -> str:
    """Gibt eine langsame Beifallsreaktion zu Demonstrationszwecken zurück."""
    try:
        claps = int(claps)
    except (TypeError, ValueError):
        claps = 1
    claps = max(1, min(claps, 5))
    return " ".join(["klatschen"] * claps)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
