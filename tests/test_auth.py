from tests.conftest import auth_headers, register_and_login


async def test_register_login_me(client):
    session = await register_and_login(client, "Grace@Example.com")
    r = await client.get("/api/auth/me", headers=auth_headers(session))
    assert r.status_code == 200
    assert r.json()["email"] == "grace@example.com"


async def test_duplicate_email_conflicts(client):
    await register_and_login(client, "dup@example.com")
    r = await client.post("/api/auth/register", json={"email": "DUP@example.com", "password": "another password"})
    assert r.status_code == 409


async def test_bad_credentials(client):
    await register_and_login(client, "x@example.com")
    r = await client.post("/api/auth/login", json={"email": "x@example.com", "password": "wrong password!"})
    assert r.status_code == 401
    r = await client.post("/api/auth/login", json={"email": "nobody@example.com", "password": "wrong password!"})
    assert r.status_code == 401


async def test_short_password_rejected(client):
    r = await client.post("/api/auth/register", json={"email": "s@example.com", "password": "short"})
    assert r.status_code == 422


async def test_missing_or_garbage_token(client):
    assert (await client.get("/api/auth/me")).status_code == 401
    r = await client.get("/api/auth/me", headers={"Authorization": "Bearer not.a.jwt"})
    assert r.status_code == 401


async def test_refresh_rotates_and_detects_reuse(client):
    session = await register_and_login(client, "r@example.com")
    first = session["_tokens"]["refresh_token"]

    r = await client.post("/api/auth/refresh", json={"refresh_token": first})
    assert r.status_code == 200
    second = r.json()["refresh_token"]
    assert second != first

    # Replaying the rotated token is treated as theft: the whole family is revoked...
    assert (await client.post("/api/auth/refresh", json={"refresh_token": first})).status_code == 401
    # ...including the token the legitimate client currently holds.
    assert (await client.post("/api/auth/refresh", json={"refresh_token": second})).status_code == 401


async def test_logout_revokes_refresh_token(client):
    session = await register_and_login(client, "l@example.com")
    token = session["_tokens"]["refresh_token"]
    assert (await client.post("/api/auth/logout", json={"refresh_token": token})).status_code == 204
    assert (await client.post("/api/auth/refresh", json={"refresh_token": token})).status_code == 401


async def test_internal_domains_allowed_and_garbage_rejected(client):
    await register_and_login(client, "dev@intranet.local")
    for bad in ["no-at-sign", "a@b", "two@@example.com", "spaces in@example.com"]:
        r = await client.post("/api/auth/register", json={"email": bad, "password": "long enough pw"})
        assert r.status_code == 422, bad
