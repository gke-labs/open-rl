"""Which client sessions are using which owners.

An owner is the scheduler's ownerID, the trainer and sampler pair that
serves one FFT job or one LoRA base model. The tinker client heartbeats its
session every ten seconds for as long as it runs. A session that stops is
dead, and an owner whose sessions are all dead is abandoned, so a job that
exits without calling delete_model still gives its GPUs back.

Everything lives in the store, so an API server restart keeps it:

  open_rl:session:<id>    present while the session is live. Each heartbeat
                          resets its expiry, so a silent session vanishes
                          on its own.
  open_rl:owner:<owner>   the sessions using this owner's workers.
  open_rl:owners          every owner that has workers.

Nothing here is atomic. The API server holds a lock per owner around attach and
around the in_use check and the teardown that follows it, so a session cannot
attach to an owner between the check and the delete.
"""

from server.store import StateStore


class SessionRegistry:
  def __init__(self, state: StateStore, ttl_seconds: float = 120.0):
    self.state = state
    self.ttl_seconds = ttl_seconds

  async def heartbeat(self, session_id: str) -> None:
    await self.state.set_value(f"open_rl:session:{session_id}", "1", ttl_seconds=self.ttl_seconds)

  async def live(self, session_id: str) -> bool:
    return await self.state.get_value(f"open_rl:session:{session_id}") is not None

  async def attach(self, session_id: str, owner: str) -> None:
    """The session is using this owner's workers from now on."""
    await self.heartbeat(session_id)
    await self.state.add_to_set("open_rl:owners", owner)
    await self.state.add_to_set(f"open_rl:owner:{owner}", session_id)

  async def owners(self) -> list[str]:
    return sorted(await self.state.set_members("open_rl:owners"))

  async def in_use(self, owner: str) -> bool:
    """Whether any of the owner's sessions is still live. Drops the dead ones."""
    for session_id in await self.state.set_members(f"open_rl:owner:{owner}"):
      if not await self.live(session_id):
        await self.state.remove_from_set(f"open_rl:owner:{owner}", session_id)
    return bool(await self.state.set_members(f"open_rl:owner:{owner}"))

  async def forget(self, owner: str) -> None:
    """The owner's workers are gone."""
    await self.state.delete_values(f"open_rl:owner:{owner}")
    await self.state.remove_from_set("open_rl:owners", owner)
