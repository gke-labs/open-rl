"""The skill is a plain Agent Skills directory; this keeps its frontmatter loadable."""

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class SkillDocsTest(unittest.TestCase):
  def test_skill_frontmatter_names_the_skill(self):
    text = (ROOT / "skills" / "openrl" / "SKILL.md").read_text()
    self.assertTrue(text.startswith("---\nname: openrl\n"), "SKILL.md needs the name: openrl frontmatter")
    self.assertIn("description:", text.split("---")[1])

  def test_the_skill_does_not_carry_a_copy_of_the_settings(self):
    # The settings are described once, in the server's OpenAPI document. A table
    # here would drift; the reference sends agents to the live schema instead.
    text = (ROOT / "skills" / "openrl" / "references" / "settings.md").read_text()
    self.assertIn("components.schemas.OpenRLSettings", text)
    self.assertNotIn("| `openrl.", text, "no settings table in the skill; read it from the server")


if __name__ == "__main__":
  unittest.main()
