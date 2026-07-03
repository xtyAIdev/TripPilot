from trip_pilot.tools.gaode_mcp_tools import get_weather_via_gaode, search_poi_via_gaode
from trip_pilot.tools.hotel_mcp_tools import search_hotels_via_mcp


def main():
    """小规模测试 MCP 调用，不走完整 Agent。"""
    print("测试高德天气：")
    print(get_weather_via_gaode.invoke({"city": "杭州"}))

    print("\n测试高德 POI：")
    print(search_poi_via_gaode.invoke({"keyword": "西湖", "city": "杭州"})[:1200])

    print("\n测试酒店搜索：")
    print(
        search_hotels_via_mcp.invoke(
            {
                "city": "杭州",
                "stay_nights": 1,
                "adult_count": 2,
                "max_price_per_night": 400,
                "size": 3,
            }
        )[:1200]
    )


if __name__ == "__main__":
    main()

