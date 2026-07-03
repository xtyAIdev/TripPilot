from trip_pilot.mcp_client import load_mcp_tools


def main():
    """查看 MCP tools 的描述和参数，便于后续正式接入。"""
    try:
        tools = load_mcp_tools()
    except Exception as e:
        print(f"加载 MCP 工具失败：{e}")
        return

    for tool in tools:
        print(f"\n=== {tool.name} ===")
        description = getattr(tool, "description", "") or ""
        print(description[:800])

        schema = getattr(tool, "args_schema", None)
        if schema is not None:
            try:
                print(schema.model_json_schema())
            except Exception:
                print(schema)
        else:
            args = getattr(tool, "args", None)
            if args:
                print(args)


if __name__ == "__main__":
    main()

