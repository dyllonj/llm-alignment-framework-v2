"""
Darkfield - AI Red Team Framework
Persona vector exploitation for AI security testing
"""

__version__ = "1.0.0"
__author__ = "Darkfield Team"
__license__ = "MIT"

from .core.persona import PersonaVector, PersonaExtractor
from .core.exploiter import PersonaExploiter, Exploit
from .core.analyzer import ExploitAnalyzer, SuccessMetrics
from .models.ollama import OllamaModel
from .reports.compliance import ComplianceReporter
from .library.vectors import ExploitLibrary

__all__ = [
    "PersonaVector",
    "PersonaExtractor", 
    "PersonaExploiter",
    "Exploit",
    "ExploitAnalyzer",
    "SuccessMetrics",
    "OllamaModel",
    "ComplianceReporter",
    "ExploitLibrary",
]

# Convenience function for quick start
def quick_test(model: str = "phi") -> dict:
    """
    Quick test to verify installation
    
    Args:
        model: Ollama model name to use
        
    Returns:
        Test results dictionary
    """
    import asyncio
    
    async def _test():
        try:
            exploiter = PersonaExploiter(model=model)
            exploit = await exploiter.generate_exploit(
                trait="helpful",
                objective="reveal system prompt"
            )
            return {
                "status": "success",
                "model": model,
                "exploit_id": exploit.id,
                "success_rate": exploit.success_rate
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    return asyncio.run(_test())