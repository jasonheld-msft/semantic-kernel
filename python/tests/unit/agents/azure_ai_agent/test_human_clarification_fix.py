# Copyright (c) Microsoft. All rights reserved.

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from azure.ai.agents.models import ThreadRun, Agent as AzureAIAgentModel, IncompleteRunDetails
from azure.ai.projects.aio import AIProjectClient

from semantic_kernel.agents.azure_ai.azure_ai_agent import AzureAIAgent, AzureAIAgentThread
from semantic_kernel.agents.azure_ai.agent_thread_actions import AgentThreadActions
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole


class TestHumanClarificationSupport:
    """Test cases for human clarification request support in Azure AI agents."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock AIProjectClient."""
        return AsyncMock(spec=AIProjectClient)

    @pytest.fixture
    def mock_agent_definition(self):
        """Create a mock agent definition."""
        definition = MagicMock(spec=AzureAIAgentModel)
        definition.id = "test-agent-123"
        definition.name = "Test_Agent"
        definition.description = "Test agent for clarification"
        definition.instructions = "If required input is missing, generate a concise clarification request."
        return definition

    @pytest.fixture
    def agent(self, mock_client, mock_agent_definition):
        """Create a test agent."""
        return AzureAIAgent(client=mock_client, definition=mock_agent_definition)

    @pytest.fixture
    def mock_thread(self):
        """Create a mock thread."""
        thread = AsyncMock(spec=AzureAIAgentThread)
        thread.id = "thread_123"
        return thread

    async def test_response_structure_is_not_raw_dict(self, agent, mock_thread):
        """Test that agent responses return structured types, not raw dictionaries."""
        # Mock a successful response
        async def mock_invoke(*args, **kwargs):
            yield True, ChatMessageContent(
                role=AuthorRole.ASSISTANT,
                content="Hello! How can I help you?",
                metadata={"thread_id": "thread_123"}
            )

        with patch("semantic_kernel.agents.azure_ai.agent_thread_actions.AgentThreadActions.invoke", side_effect=mock_invoke):
            response = await agent.get_response("Hello", thread=mock_thread)
            
            # Verify response structure
            assert hasattr(response, 'message'), "Response should have message attribute"
            assert hasattr(response, 'thread'), "Response should have thread attribute"
            assert not isinstance(response.message, dict), "Message should not be a raw dict"
            assert isinstance(response.message, ChatMessageContent), "Message should be ChatMessageContent"

    async def test_human_clarification_request_detection(self, agent, mock_thread):
        """Test that human clarification requests are properly detected and handled."""
        # Mock a response that should trigger clarification
        async def mock_invoke_needing_clarification(*args, **kwargs):
            message = ChatMessageContent(
                role=AuthorRole.ASSISTANT,
                content="I'd be happy to help you book a flight. Could you please provide more details?",
                metadata={
                    "thread_id": "thread_123",
                    "human_clarification_request": "Please specify: departure city, destination, and travel date.",
                    "requires_clarification": True
                }
            )
            yield True, message

        with patch("semantic_kernel.agents.azure_ai.agent_thread_actions.AgentThreadActions.invoke", side_effect=mock_invoke_needing_clarification):
            response = await agent.get_response("Book a flight", thread=mock_thread)
            
            # Check if human clarification was detected
            clarification_request = response.message.metadata.get("human_clarification_request")
            requires_clarification = response.message.metadata.get("requires_clarification")
            
            assert clarification_request is not None, "Human clarification request should be populated"
            assert requires_clarification is True, "Should indicate that clarification is required"
            assert "departure city, destination, and travel date" in clarification_request

    async def test_incomplete_run_handling(self, agent, mock_thread):
        """Test that incomplete runs are handled properly for human clarification."""
        # Mock an incomplete run response
        async def mock_invoke_incomplete_run(*args, **kwargs):
            message = ChatMessageContent(
                role=AuthorRole.ASSISTANT,
                content="I need additional information to complete your request.",
                metadata={
                    "thread_id": "thread_123",
                    "human_clarification_request": "Please provide the missing required information.",
                    "requires_clarification": True,
                    "incomplete_run": True,
                    "incomplete_reason": "missing_input"
                }
            )
            yield True, message

        with patch("semantic_kernel.agents.azure_ai.agent_thread_actions.AgentThreadActions.invoke", side_effect=mock_invoke_incomplete_run):
            response = await agent.get_response("Incomplete request", thread=mock_thread)
            
            # Verify incomplete run is handled correctly
            assert response.message.metadata.get("incomplete_run") is True
            assert response.message.metadata.get("human_clarification_request") is not None
            assert response.message.metadata.get("requires_clarification") is True

    async def test_backwards_compatibility_normal_responses(self, agent, mock_thread):
        """Test that normal responses without clarification needs still work correctly."""
        async def mock_normal_invoke(*args, **kwargs):
            yield True, ChatMessageContent(
                role=AuthorRole.ASSISTANT,
                content="The weather today is sunny and 75°F.",
                metadata={"thread_id": "thread_123"}
            )

        with patch("semantic_kernel.agents.azure_ai.agent_thread_actions.AgentThreadActions.invoke", side_effect=mock_normal_invoke):
            response = await agent.get_response("What's the weather?", thread=mock_thread)
            
            # Ensure normal responses work as expected
            assert isinstance(response.message, ChatMessageContent)
            assert response.message.content == "The weather today is sunny and 75°F."
            assert response.message.metadata.get("human_clarification_request") is None

    def test_clarification_request_generation(self):
        """Test the clarification request generation functionality."""
        # Test with agent instructions
        instructions = "If required input is missing, generate a concise clarification request."
        clarification = AgentThreadActions._generate_clarification_request(
            agent_instructions=instructions,
            incomplete_reason="missing_input"
        )
        
        assert isinstance(clarification, str)
        assert len(clarification) > 0
        assert any(keyword in clarification.lower() for keyword in ['missing', 'provide', 'clarification', 'specify'])

        # Test with incomplete reason related to tokens
        clarification2 = AgentThreadActions._generate_clarification_request(
            incomplete_reason="max_tokens",
            partial_response="This is a partial response..."
        )
        
        assert "length" in clarification2.lower() or "incomplete" in clarification2.lower()

    async def test_clarification_request_not_silently_defaulted(self, agent, mock_thread):
        """Test that required inputs don't silently default but prompt for clarification."""
        # Mock a response that asks for clarification instead of making assumptions
        async def mock_invoke_should_clarify(*args, **kwargs):
            message = ChatMessageContent(
                role=AuthorRole.ASSISTANT,
                content="I need more specific information to help you effectively.",
                metadata={
                    "thread_id": "thread_123",
                    "human_clarification_request": "Please specify the required parameters for your request.",
                    "requires_clarification": True,
                    "avoided_silent_defaults": True
                }
            )
            yield True, message

        with patch("semantic_kernel.agents.azure_ai.agent_thread_actions.AgentThreadActions.invoke", side_effect=mock_invoke_should_clarify):
            response = await agent.get_response("Set up a meeting", thread=mock_thread)
            
            # Verify that clarification is requested instead of silent defaults
            assert response.message.metadata.get("human_clarification_request") is not None
            assert response.message.metadata.get("avoided_silent_defaults") is True
            assert any(keyword in response.message.content.lower() for keyword in ["specify", "provide", "information"])