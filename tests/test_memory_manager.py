import unittest

from memory.memory_manager import MemoryManager


class TestMemoryManager(unittest.TestCase):

    def test_add_and_retrieve_semantic_memory(self):
        manager = MemoryManager(memory_types=["working", "episodic", "semantic"])
        memory_id = manager.add_memory(
            content="RAG 可以减少幻觉",
            memory_type="semantic",
            importance=0.8
        )
        self.assertTrue(memory_id)

        results = manager.retrieve_memories(
            query="减少幻觉",
            memory_types=["semantic"]
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].memory_type, "semantic")

    def test_forget_low_importance(self):
        manager = MemoryManager(memory_types=["working"])
        manager.add_memory(content="low", memory_type="working", importance=0.05)
        manager.add_memory(content="high", memory_type="working", importance=0.9)

        removed = manager.forget_memories(threshold=0.1)
        self.assertEqual(removed, 1)

        results = manager.retrieve_memories(query="", memory_types=["working"])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].content, "high")


if __name__ == "__main__":
    unittest.main()
