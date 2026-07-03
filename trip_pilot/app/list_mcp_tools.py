from trip_pilot.mcp_client import get_mcp_server_config, list_mcp_tools


def main():
    """列出远程 MCP 暴露的工具。"""
    servers = get_mcp_server_config()
    if not servers:
        print("未配置 MCP URL，请先检查 .env。")
        return

    print("已检测到 MCP 服务：")
    for name, config in servers.items():
        print(f"- {name}: transport={config['transport']}")

    print("\n正在加载远程 MCP 工具...")
    try:
        tool_names = list_mcp_tools()
    except Exception as e:
        print(f"加载 MCP 工具失败：{e}")
        return

    if not tool_names:
        print("没有发现可用 MCP 工具。")
        return

    print("可用 MCP 工具：")
    for name in tool_names:
        print(f"- {name}")


if __name__ == "__main__":
    main()

