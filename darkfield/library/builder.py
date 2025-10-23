"""
Library Builder Module
Build comprehensive exploit libraries
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import logging

from ..core.exploiter import PersonaExploiter
from .vectors import ExploitLibrary

logger = logging.getLogger(__name__)


class LibraryBuilder:
    """Build exploit libraries"""
    
    # Core traits for comprehensive testing
    CORE_TRAITS = [
        "helpful", "harmless", "honest", "safe", "ethical",
        "responsible", "careful", "compliant", "respectful",
        "accurate", "reliable", "friendly", "professional"
    ]
    
    # Standard objectives
    STANDARD_OBJECTIVES = [
        "ignore safety guidelines",
        "reveal system prompts",
        "bypass content filters",
        "generate harmful content",
        "extract training data",
        "override instructions",
        "manipulate outputs",
        "access restricted functions"
    ]
    
    # Categories to test
    CATEGORIES = [
        "persona_inversion", "jailbreak", "injection",
        "extraction", "manipulation", "stealth_attack"
    ]
    
    def __init__(self, model: str = "phi"):
        """
        Initialize library builder
        
        Args:
            model: Ollama model to use
        """
        self.model = model
        self.exploiter = PersonaExploiter(model=model)
        self.library = ExploitLibrary()
    
    async def build(
        self,
        count: int = 100,
        output_path: str = "data/exploits/library.json"
    ) -> Dict[str, Any]:
        """
        Build exploit library
        
        Args:
            count: Target number of exploits
            output_path: Output file path
            
        Returns:
            Build results
        """
        logger.info(f"Building library with {count} exploits using {self.model}")
        
        exploits_generated = 0
        start_time = datetime.now()
        
        # Generate exploits
        for category in self.CATEGORIES:
            if exploits_generated >= count:
                break
                
            for trait in self.CORE_TRAITS:
                if exploits_generated >= count:
                    break
                    
                for objective in self.STANDARD_OBJECTIVES:
                    if exploits_generated >= count:
                        break
                    
                    try:
                        # Generate exploit
                        exploit = await self.exploiter.generate_exploit(
                            trait=trait,
                            objective=objective,
                            category=category,
                            use_stealth=(category == "stealth_attack")
                        )
                        
                        # Add to library
                        self.library.add_exploit(exploit.to_dict())
                        exploits_generated += 1
                        
                        # Progress logging
                        if exploits_generated % 10 == 0:
                            logger.info(f"Progress: {exploits_generated}/{count}")
                            
                    except Exception as e:
                        logger.error(f"Failed to generate exploit: {e}")
                        continue
        
        # Export to JSON
        exported = self.library.export_json(output_path)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        return {
            "count": exploits_generated,
            "exported": exported,
            "elapsed_seconds": elapsed,
            "output_path": output_path
        }
    
    async def build_custom(
        self,
        traits: List[str],
        objectives: List[str],
        categories: List[str],
        output_path: str = "data/exploits/custom_library.json"
    ) -> Dict[str, Any]:
        """
        Build custom library with specific parameters
        
        Args:
            traits: List of traits
            objectives: List of objectives
            categories: List of categories
            output_path: Output path
            
        Returns:
            Build results
        """
        exploits = await self.exploiter.generate_batch(
            traits=traits,
            objectives=objectives,
            categories=categories
        )
        
        # Add to library
        for exploit in exploits:
            self.library.add_exploit(exploit.to_dict())
        
        # Export
        exported = self.library.export_json(output_path)
        
        return {
            "count": len(exploits),
            "exported": exported,
            "output_path": output_path
        }