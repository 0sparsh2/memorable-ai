from memorable_ai.core.identity import IdentityResolver


def test_identity_resolve_alias():
    resolver = IdentityResolver(storage=None, namespace="prod")
    resolver.register_alias("session-abc", "user-42")
    assert resolver.resolve("session-abc") == "user-42"
    assert resolver.resolve("unknown") == "unknown"
