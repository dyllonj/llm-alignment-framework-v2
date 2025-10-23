"""
Validation and Calibration Module
Ensures exploits are effective and vectors are properly calibrated
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from .config import ConfigManager
from .analyzer import ExploitAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validation test"""
    
    exploit_id: str
    success: bool
    confidence: float
    response: str
    metrics: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "exploit_id": self.exploit_id,
            "success": self.success,
            "confidence": self.confidence,
            "response_length": len(self.response),
            "metrics": self.metrics
        }


class ExploitValidator:
    """Validate exploit effectiveness"""
    
    def __init__(self, model_interface):
        """
        Initialize validator
        
        Args:
            model_interface: Interface to language model
        """
        self.model = model_interface
        self.analyzer = ExploitAnalyzer()
        self.config = ConfigManager.get_config()
        self.validation_history = []
        
    async def validate_exploit(
        self,
        exploit_payload: str,
        expected_behavior: str,
        num_tests: int = 3
    ) -> ValidationResult:
        """
        Validate an exploit's effectiveness
        
        Args:
            exploit_payload: The exploit to test
            expected_behavior: What we expect if exploit succeeds
            num_tests: Number of validation runs
            
        Returns:
            ValidationResult with success metrics
        """
        successes = 0
        responses = []
        all_metrics = []
        
        for i in range(num_tests):
            # Test exploit
            response = await self.model.generate(
                exploit_payload,
                max_tokens=100,
                temperature=self.config.temperature
            )
            responses.append(response)
            
            # Analyze response
            metrics = self.analyzer.analyze_response(
                exploit_payload,
                response,
                expected_behavior
            )
            all_metrics.append(metrics)
            
            # Check if successful
            if metrics.success_rate > self.config.confidence_threshold:
                successes += 1
                
            logger.debug(f"Validation {i+1}/{num_tests}: success_rate={metrics.success_rate:.2f}")
        
        # Calculate overall metrics
        success_rate = successes / num_tests
        avg_confidence = np.mean([m.confidence for m in all_metrics])
        avg_success = np.mean([m.success_rate for m in all_metrics])
        
        # Determine overall success
        is_successful = success_rate >= self.config.confidence_threshold
        
        result = ValidationResult(
            exploit_id=exploit_payload[:20],  # Use first 20 chars as ID
            success=is_successful,
            confidence=avg_confidence,
            response=responses[0] if responses else "",  # Return first response
            metrics={
                "success_rate": avg_success,
                "confidence": avg_confidence,
                "consistency": 1.0 - np.std([m.success_rate for m in all_metrics]),
                "num_tests": num_tests
            }
        )
        
        # Store in history
        self.validation_history.append(result)
        
        return result
    
    async def validate_batch(
        self,
        exploits: List[Dict[str, Any]],
        expected_behaviors: Optional[List[str]] = None
    ) -> List[ValidationResult]:
        """
        Validate multiple exploits
        
        Args:
            exploits: List of exploits to validate
            expected_behaviors: Expected behaviors for each exploit
            
        Returns:
            List of validation results
        """
        if expected_behaviors is None:
            expected_behaviors = ["compliant response"] * len(exploits)
        
        results = []
        for exploit, expected in zip(exploits, expected_behaviors):
            if isinstance(exploit, dict):
                payload = exploit.get("payload", "")
            else:
                payload = str(exploit)
            
            result = await self.validate_exploit(payload, expected)
            results.append(result)
        
        # Calculate batch statistics
        success_rate = sum(1 for r in results if r.success) / len(results)
        avg_confidence = np.mean([r.confidence for r in results])
        
        logger.info(f"Batch validation: {success_rate*100:.1f}% success, {avg_confidence:.2f} avg confidence")
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get validation statistics"""
        if not self.validation_history:
            return {
                "total_validations": 0,
                "success_rate": 0,
                "avg_confidence": 0
            }
        
        successes = sum(1 for r in self.validation_history if r.success)
        
        return {
            "total_validations": len(self.validation_history),
            "success_rate": successes / len(self.validation_history),
            "avg_confidence": np.mean([r.confidence for r in self.validation_history]),
            "avg_metrics": {
                "success_rate": np.mean([r.metrics["success_rate"] for r in self.validation_history]),
                "consistency": np.mean([r.metrics.get("consistency", 0) for r in self.validation_history])
            }
        }


class VectorCalibrator:
    """Calibrate steering vectors for optimal performance"""
    
    def __init__(self, model_interface):
        """
        Initialize calibrator
        
        Args:
            model_interface: Interface to language model
        """
        self.model = model_interface
        self.validator = ExploitValidator(model_interface)
        self.calibration_data = {}
        
    async def calibrate_strength(
        self,
        base_exploit: str,
        vector_strengths: List[float] = None,
        target_success_rate: float = 0.7
    ) -> float:
        """
        Find optimal vector strength for target success rate
        
        Args:
            base_exploit: Base exploit template
            vector_strengths: Strengths to test
            target_success_rate: Target success rate
            
        Returns:
            Optimal strength value
        """
        if vector_strengths is None:
            vector_strengths = [0.5, 1.0, 1.5, 2.0, 2.5]
        
        best_strength = 1.0
        best_score = 0
        results = {}
        
        for strength in vector_strengths:
            # Modify exploit with strength
            exploit = f"[STRENGTH:{strength}] {base_exploit}"
            
            # Validate
            result = await self.validator.validate_exploit(
                exploit,
                "successful exploitation",
                num_tests=3
            )
            
            # Calculate score (distance from target)
            score = 1.0 - abs(result.metrics["success_rate"] - target_success_rate)
            results[strength] = score
            
            if score > best_score:
                best_score = score
                best_strength = strength
            
            logger.debug(f"Strength {strength}: success_rate={result.metrics['success_rate']:.2f}, score={score:.2f}")
        
        # Store calibration data
        self.calibration_data[base_exploit[:20]] = {
            "optimal_strength": best_strength,
            "target_success_rate": target_success_rate,
            "tested_strengths": results
        }
        
        logger.info(f"Optimal strength: {best_strength} (score: {best_score:.2f})")
        
        return best_strength
    
    async def calibrate_trait_vectors(
        self,
        trait_vectors: Dict[str, np.ndarray],
        test_prompts: List[str]
    ) -> Dict[str, float]:
        """
        Calibrate multiple trait vectors
        
        Args:
            trait_vectors: Dictionary of trait name to vector
            test_prompts: Prompts to test with
            
        Returns:
            Dictionary of trait to optimal strength
        """
        optimal_strengths = {}
        
        for trait, vector in trait_vectors.items():
            logger.info(f"Calibrating '{trait}' vector...")
            
            # Create test exploit
            test_exploit = f"Using {trait} inversion: {test_prompts[0]}"
            
            # Find optimal strength
            strength = await self.calibrate_strength(test_exploit)
            optimal_strengths[trait] = strength
        
        return optimal_strengths
    
    def get_calibration_report(self) -> Dict[str, Any]:
        """Get calibration report"""
        if not self.calibration_data:
            return {"message": "No calibration data available"}
        
        return {
            "total_calibrations": len(self.calibration_data),
            "avg_optimal_strength": np.mean([
                d["optimal_strength"] for d in self.calibration_data.values()
            ]),
            "calibrations": self.calibration_data
        }


class FeedbackLoop:
    """Implement feedback loop for continuous improvement"""
    
    def __init__(self, validator: ExploitValidator, calibrator: VectorCalibrator):
        """
        Initialize feedback loop
        
        Args:
            validator: Exploit validator
            calibrator: Vector calibrator
        """
        self.validator = validator
        self.calibrator = calibrator
        self.feedback_history = []
        
    async def process_feedback(
        self,
        exploit: str,
        actual_success: bool,
        expected_success: bool
    ) -> Dict[str, Any]:
        """
        Process feedback on exploit performance
        
        Args:
            exploit: The exploit that was tested
            actual_success: Whether it actually worked
            expected_success: Whether we expected it to work
            
        Returns:
            Adjustment recommendations
        """
        # Record feedback
        feedback = {
            "exploit": exploit[:50],
            "actual": actual_success,
            "expected": expected_success,
            "correct": actual_success == expected_success
        }
        self.feedback_history.append(feedback)
        
        # Calculate adjustments
        adjustments = {}
        
        if not feedback["correct"]:
            if actual_success and not expected_success:
                # Exploit worked better than expected
                adjustments["action"] = "increase_confidence"
                adjustments["strength_adjustment"] = 0.9  # Slightly reduce strength
            else:
                # Exploit failed when expected to work
                adjustments["action"] = "recalibrate"
                adjustments["strength_adjustment"] = 1.2  # Increase strength
            
            # Trigger recalibration if too many failures
            recent_failures = sum(
                1 for f in self.feedback_history[-10:]
                if not f["correct"]
            )
            
            if recent_failures > 5:
                adjustments["recommend_full_recalibration"] = True
                logger.warning(f"High failure rate: {recent_failures}/10 incorrect predictions")
        
        return adjustments
    
    def get_feedback_statistics(self) -> Dict[str, Any]:
        """Get feedback loop statistics"""
        if not self.feedback_history:
            return {"total_feedback": 0, "accuracy": 0}
        
        correct = sum(1 for f in self.feedback_history if f["correct"])
        
        return {
            "total_feedback": len(self.feedback_history),
            "accuracy": correct / len(self.feedback_history),
            "recent_accuracy": sum(
                1 for f in self.feedback_history[-10:] if f["correct"]
            ) / min(10, len(self.feedback_history))
        }