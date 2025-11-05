"""
AGENTS.md Loader Module

Implements the discovery hierarchy and parsing logic for AGENTS.md files
as defined in agents_md_framework.md
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import re


class AgentsMDLoader:
    """
    Loads and parses AGENTS.md files following the standard discovery hierarchy:
    1. ./AGENTS.md in current working directory
    2. Nearest parent directory up to repo root
    3. AGENTS.md in subdirectories the agent is editing
    4. Personal override: ~/.factory/AGENTS.md
    """

    def __init__(self, working_dir: Optional[Path] = None):
        self.working_dir = working_dir or Path.cwd()

    def discover_agents_md(self, start_path: Optional[Path] = None) -> Optional[Path]:
        """
        Discover AGENTS.md file following the standard hierarchy.

        Returns the path to the first AGENTS.md file found, or None.
        """
        search_path = start_path or self.working_dir

        # Search upward from current directory to root
        current = search_path.resolve()
        while current != current.parent:
            agents_md = current / "AGENTS.md"
            if agents_md.exists():
                return agents_md
            current = current.parent

        # Check root
        agents_md = current / "AGENTS.md"
        if agents_md.exists():
            return agents_md

        # Check personal override
        personal_override = Path.home() / ".factory" / "AGENTS.md"
        if personal_override.exists():
            return personal_override

        return None

    def load_agents_md(self, path: Path) -> Dict[str, Any]:
        """
        Load and parse an AGENTS.md file.

        Returns a structured dictionary with agent instructions, conventions,
        tools, and metadata.
        """
        if not path.exists():
            raise FileNotFoundError(f"AGENTS.md not found at {path}")

        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        return self._parse_agents_md(content)

    def _parse_agents_md(self, content: str) -> Dict[str, Any]:
        """
        Parse AGENTS.md content into structured data.

        Extracts sections like:
        - Project Overview
        - Agent Role
        - Conventions & Patterns
        - Available Tools
        - Example Tasks
        """
        sections = {}
        current_section = None
        current_content = []

        for line in content.split('\n'):
            # Detect section headers (## Section Name)
            header_match = re.match(r'^##\s+(.+)$', line)
            if header_match:
                # Save previous section
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()

                # Start new section
                current_section = header_match.group(1).strip()
                current_content = []
            else:
                if current_section:
                    current_content.append(line)

        # Save last section
        if current_section:
            sections[current_section] = '\n'.join(current_content).strip()

        return {
            'raw_content': content,
            'sections': sections,
            'instructions': sections.get('Agent Instructions', sections.get('Overview', '')),
            'conventions': sections.get('Conventions & Patterns', ''),
            'tools': self._extract_tools(sections.get('Available Tools', '')),
            'example_tasks': sections.get('Example Tasks', ''),
        }

    def _extract_tools(self, tools_section: str) -> list:
        """Extract list of tools from the Available Tools section"""
        tools = []
        for line in tools_section.split('\n'):
            line = line.strip()
            if line.startswith('-') or line.startswith('*'):
                tool_name = line.lstrip('-*').strip()
                if tool_name:
                    tools.append(tool_name)
        return tools

    def get_agent_instructions(self, path: Path) -> str:
        """
        Get the complete agent instructions from AGENTS.md.
        This is the core prompt that will be sent to the LLM.
        """
        parsed = self.load_agents_md(path)
        return parsed['raw_content']
