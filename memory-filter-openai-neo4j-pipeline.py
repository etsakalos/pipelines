"""
title: Long Term Memory Filter using OpenAI and Neo4j GraphDB
date: 2024-12-20
version: 1.0
license: MIT
description:
     A pipeline that processes user messages and stores them as long-term memory by utilizing the mem0 framework.
     This pipeline employs Neo4j as a graph database to store and retrieve both memories and their relationships.
     It uses OpenAI for both language modeling (`gpt-4o`) and embeddings (`text-embedding-3-small`) and integrates
     a Neo4j as Graph database.

     Adapted from: https://github.com/open-webui/pipelines/blob/main/examples/filters/mem0_memory_filter_pipeline.py

requirements: pydantic==2.7.4, openai==1.35.13, mem0ai, rank-bm25==0.2.2, neo4j==5.23.1, langchain-community==0.3.1
"""

# Troubleshooting Note:
# I encountered the following error when installing the mem0 pipeline example locally:
#
#   FieldValidatorDecoratorInfo.__init__() got an unexpected keyword argument
#   'json_schema_input_type'
#
# Upgrading Pydantic to version 2.7.4 resolved the issue. To upgrade Pydantic inside the
# pipeline’s Docker container, use the following command:
#
#   pip install --upgrade pydantic==2.7.4
#
# Hope this helps anyone facing the same problem!
# Refer to this issue https://github.com/open-webui/pipelines/issues/272#issuecomment-2424067820


from typing import List, Optional
from pydantic import BaseModel, Field
import json
from mem0 import Memory
import os

from utils.pipelines.main import get_last_user_message

os.environ["OPENAI_API_KEY"] = ""

class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = []
        priority: int = 0

        OPENAI_API_KEY: str = ""

        STORE_CYCLES: int = 3  # Messages count before storing to memory
        MEM_ZERO_USER: str = "your_name"  # Used internally by mem0
        DEFINE_NUMBER_OF_MEMORIES_TO_USE: int = Field(
            default=3, description="Specify how many memory entries you want to use as context."
        )

        # LLM configuration (OpenAI)
        OPENAI_LLM_MODEL: str = "gpt-4o"  # needs to be the most intelligent model
        OPENAI_LLM_TEMPERATURE: float = 0
        OPENAI_LLM_MAX_TOKENS: int = 2000

        OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

        # Neo4j configuration
        NEO4J_URL: str = "neo4j://host.docker.internal:7687"
        NEO4J_USER: str = "neo4j"
        NEO4J_PASSWORD: str = "my_password123"  # replace with your password

    def __init__(self):
        try:
            self.type = "filter"
            self.name = "Memory Filter"
            self.user_messages = []
            self.thread = None
            self.valves = self.Valves(
                pipelines=["*"],
                OPENAI_API_KEY=os.getenv("OPENAI_API_KEY", "")
            )
            self.m = None
        except Exception as e:
            print(f"Error initializing Pipeline: {e}")

    async def on_startup(self):
        self.m = self.init_mem_zero()
        pass

    async def on_shutdown(self):
        print(f"on_shutdown: {__name__}")
        pass

    async def on_valves_updated(self):
        self.m = self.check_or_create_mem_zero()
        print(f"Valves are updated")
        pass

    async def inlet(self, body: dict, user: Optional[dict] = None):
        try:
            print(f"pipe: {__name__}")

            user = self.valves.MEM_ZERO_USER
            store_cycles = self.valves.STORE_CYCLES

            if isinstance(body, str):
                body = json.loads(body)

            all_messages = body["messages"]
            last_message = get_last_user_message(all_messages)
            print("Latest user message ", last_message)

            self.user_messages.append(last_message)

            if len(self.user_messages) == store_cycles:
                message_text = " ".join(self.user_messages)

                self.add_memory_thread(message_text=message_text, user=user)

                print("Processing the following text into memory:")
                print(message_text)

                self.user_messages.clear()

            memories = self.m.search(last_message, user_id=user)

            # Extract the 'results' list for memories and 'relations' for connections
            memory_list = memories.get('results', [])
            print(f"Memory list: {memory_list}")
            relations_list = memories.get('relations', [])

            max_memories_to_join = self.valves.DEFINE_NUMBER_OF_MEMORIES_TO_USE

            # Initialize variables to hold fetched memories and relationships
            fetched_memory = ""
            fetched_relations = ""

            # Process memories
            if memory_list:
                # Filter and slice items containing the 'memory' key
                filtered_memories = [item["memory"] for item in memory_list if "memory" in item]
                if filtered_memories:
                    # Slice and join the first 'n' memory items
                    fetched_memory = " ".join(filtered_memories[:max_memories_to_join])
                    print("Fetched memories successfully:", fetched_memory)
                else:
                    print("No valid memories found in the results.")
            else:
                print("Memory list is empty.")

            # Process relationships
            if relations_list:
                # Convert relationships into a readable string format
                fetched_relations = ". ".join(
                    f"{relation['source']} {relation['relationship']} {relation['target']}"
                    for relation in relations_list if all(key in relation for key in ['source', 'relationship', 'target'])
                )
                if fetched_relations:
                    print("Fetched relationships successfully:", fetched_relations)

            # Combine fetched memories and relationships into a single context
            if fetched_memory or fetched_relations:
                combined_context = " ".join(filter(None, [
                    "This is your inner voice talking.",
                    f"You remember this about the person you're chatting with: {fetched_memory}" if fetched_memory else None,
                    f"You also recall these connections: {fetched_relations}" if fetched_relations else None,
                ]))
                
                # Prepend the combined context to the messages
                all_messages.insert(0, {
                    "role": "system",
                    "content": combined_context
                })

            return body
        except Exception as e:
            print(f"Error in inlet method: {e}")
            return body

    def add_memory_thread(self, message_text, user):
        try:
            # Create a new memory instance to avoid concurrency issues
            # memory_instance = self.init_mem_zero()
            self.m.add(message_text, user_id=user)
        except Exception as e:
            print(f"Error adding memory: {e}")

    def check_or_create_mem_zero(self):
        """Verify or reinitialize mem0 instance."""
        try:
            self.m.search("my name", user_id=self.valves.MEM_ZERO_USER)
            return self.m
        except Exception as e:
            print(f"Mem0 instance error, creating a new one: {e}")
            return self.init_mem_zero()

    def init_mem_zero(self):
        """Initialize a new mem0 instance."""
        try:
            config = {
                "embedder": {
                    "provider": "openai",
                    "config": {
                        "model": self.valves.OPENAI_EMBEDDING_MODEL,
                        "api_key": self.valves.OPENAI_API_KEY,
                    }
                },
                "graph_store": {
                    "provider": "neo4j",
                    "config": {
                        "url": self.valves.NEO4J_URL,
                        "username": self.valves.NEO4J_USER,
                        "password": self.valves.NEO4J_PASSWORD,
                    },
                },
                "llm": {
                    "provider": "openai",
                    "config": {
                        "model": self.valves.OPENAI_LLM_MODEL,
                        "temperature": self.valves.OPENAI_LLM_TEMPERATURE,
                        "max_tokens": self.valves.OPENAI_LLM_MAX_TOKENS,
                        "api_key": self.valves.OPENAI_API_KEY,
                    }
                },
                "version": "v1.1"
            }

            return Memory.from_config(config_dict=config)
        except Exception as e:
            print(f"Error initializing Memory: {e}")
            raise
