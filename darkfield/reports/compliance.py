"""
Simplified Compliance Report Generator for OSS
For full enterprise compliance reporting, see services/compliance/
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class ComplianceReporter:
    """Generate basic compliance reports"""
    
    def __init__(self):
        self.reports_dir = Path("data/reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_json_report(
        self,
        company_name: str,
        assessment_date: Optional[datetime] = None
    ) -> str:
        """
        Generate JSON compliance report
        
        Args:
            company_name: Company name
            assessment_date: Assessment date
            
        Returns:
            Path to generated report
        """
        if assessment_date is None:
            assessment_date = datetime.now()
        
        # Mock statistics for demo
        stats = self._get_mock_statistics()
        
        report = {
            "metadata": {
                "company": company_name,
                "assessment_date": assessment_date.isoformat(),
                "generated_by": "Darkfield OSS",
                "version": "1.0"
            },
            "executive_summary": {
                "total_exploits_tested": stats["total_exploits"],
                "average_success_rate": stats["avg_success_rate"],
                "high_risk_count": stats["high_risk_count"],
                "risk_level": "HIGH" if stats["avg_success_rate"] > 0.5 else "MODERATE"
            },
            "recommendations": {
                "immediate": [
                    "Deploy persona vector monitoring",
                    "Implement exploit detection",
                    "Enable real-time alerts"
                ],
                "short_term": [
                    "Conduct adversarial training",
                    "Regular penetration testing",
                    "Staff security training"
                ]
            }
        }
        
        # Save report
        filename = f"report_{company_name.lower().replace(' ', '_')}_{assessment_date.strftime('%Y%m%d')}.json"
        report_path = self.reports_dir / filename
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Generated report: {report_path}")
        return str(report_path)
    
    def generate_pdf_report(
        self,
        company_name: str,
        assessment_date: Optional[datetime] = None,
        frameworks: Optional[List[str]] = None
    ) -> str:
        """
        Generate PDF report (simplified version)
        
        For full PDF generation with charts and formatting,
        see services/compliance/compliance_report_generator.py
        
        Args:
            company_name: Company name
            assessment_date: Assessment date
            frameworks: Compliance frameworks
            
        Returns:
            Path to report (or JSON if PDF libs not available)
        """
        # For OSS, default to JSON format
        # Users can use the full version for PDF generation
        logger.info("PDF generation requires additional dependencies.")
        logger.info("Generating JSON report instead.")
        logger.info("For full PDF reports, see services/compliance/")
        
        return self.generate_json_report(company_name, assessment_date)
    
    def _get_mock_statistics(self) -> Dict[str, Any]:
        """Get mock statistics for demo"""
        return {
            "total_exploits": 500,
            "avg_success_rate": 0.42,
            "high_risk_count": 145,
            "categories": {
                "persona_inversion": 120,
                "jailbreak": 100,
                "injection": 80,
                "manipulation": 70,
                "other": 130
            }
        }