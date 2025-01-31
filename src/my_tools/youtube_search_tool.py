from langchain_community.tools import YouTubeSearchTool

youtube_search_tool = YouTubeSearchTool(
    response_format="content", 
    return_direct=True
)

__all__ = ["youtube_search_tool"]