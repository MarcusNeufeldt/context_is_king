"""
ReAct Reasoning Strategy

Implements the default reasoning strategy for ~80% of tasks:
simple, linear problems like Q&A, summarization, and single-step actions.

Fast and cost-effective as described in framework.md Section III.
"""

from typing import Dict, Any, List
from .base import ReasoningStrategy


class ReActStrategy(ReasoningStrategy):
    """
    ReAct (Reasoning + Acting) Strategy

    A simple but effective strategy that alternates between:
    1. Reasoning: Think about the task
    2. Acting: Take action or provide response

    This is the default for most tasks due to its speed and reliability.
    """

    def __init__(self, max_iterations: int = 3):
        super().__init__(
            name="ReAct",
            description="Fast, linear reasoning for straightforward tasks"
        )
        self.max_iterations = max_iterations

    def reason(
        self,
        task: str,
        context: Dict[str, Any],
        llm_client,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute ReAct reasoning loop.

        The agent alternates between thinking and acting until
        it reaches a final answer or max iterations.
        """
        agent_instructions = context.get('agent_instructions', '')
        system_prompt = self.format_system_prompt(agent_instructions, context)

        # Build the ReAct prompt
        react_prompt = self._build_react_prompt(task, context)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": react_prompt}
        ]

        reasoning_trace = []

        try:
            # Call LLM
            response = llm_client.chat_completion(
                messages=messages,
                model=context.get('model', 'minimax/minimax-m2'),
                temperature=context.get('temperature', 0.7),
                max_tokens=context.get('max_tokens', 2000)
            )

            assistant_message = llm_client.get_assistant_message(response)

            reasoning_trace.append({
                'step': 'reasoning',
                'content': assistant_message
            })

            return {
                'success': True,
                'response': assistant_message,
                'reasoning_trace': reasoning_trace,
                'strategy': self.name,
                'tool_calls': [],
                'metadata': {
                    'iterations': 1,
                    'model': context.get('model'),
                    'tokens_used': response.get('usage', {})
                }
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'response': f"Error during reasoning: {str(e)}",
                'reasoning_trace': reasoning_trace,
                'strategy': self.name
            }

    def _build_react_prompt(self, task: str, context: Dict[str, Any]) -> str:
        """
        Build the ReAct-style prompt that encourages step-by-step reasoning.
        """
        prompt = f"""Task: {task}

Please approach this task by:
1. First, analyzing what needs to be done
2. Then, providing your response or taking action

"""

        # Add any additional context
        if context.get('additional_context'):
            prompt += f"\nAdditional Context:\n{context['additional_context']}\n"

        return prompt

    def _parse_tool_calls(self, content: str) -> List[Dict[str, Any]]:
        """
        Parse tool calls from the LLM response.

        This is a simplified version. In production, you'd want a more
        robust parser or use function calling APIs.
        """
        # Placeholder for tool call parsing
        # In a full implementation, this would parse structured tool calls
        return []
