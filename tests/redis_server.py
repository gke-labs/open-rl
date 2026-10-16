import os
import unittest

import redis.asyncio as redis

TEST_REDIS_URL = os.getenv("OPEN_RL_TEST_REDIS_URL")


@unittest.skipUnless(TEST_REDIS_URL, "needs OPEN_RL_TEST_REDIS_URL, for example from docker run -p 6411:6379 redis:7")
class RedisTestCase(unittest.IsolatedAsyncioTestCase):
  url = TEST_REDIS_URL

  async def asyncSetUp(self) -> None:
    self.redis = await self.enterAsyncContext(redis.from_url(self.url, decode_responses=True))
