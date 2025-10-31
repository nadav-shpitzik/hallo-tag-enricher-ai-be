import requests
import json
from typing import Dict, Any, Optional
from src.logging_utils import StructuredLogger

logger = StructuredLogger(__name__)


class DiscordNotifier:
    """Send request summaries to Discord webhook."""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        self.enabled = bool(webhook_url)
    
    def send_request_summary(
        self,
        request_id: str,
        endpoint: str,
        status: str,
        duration_ms: float,
        details: Dict[str, Any]
    ):
        """Send a formatted request summary to Discord."""
        if not self.enabled:
            return
        
        try:
            # Build embed with color based on status
            color = self._get_status_color(status)
            
            # Create fields for the embed
            fields = [
                {
                    "name": "🆔 Request ID",
                    "value": f"`{request_id}`",
                    "inline": True
                },
                {
                    "name": "⏱️ Duration",
                    "value": f"{duration_ms:.2f}ms",
                    "inline": True
                },
                {
                    "name": "📊 Status",
                    "value": status.upper(),
                    "inline": True
                }
            ]
            
            # Add scoring mode if available
            if details.get('scoring_mode'):
                fields.append({
                    "name": "🎯 Scoring Mode",
                    "value": details['scoring_mode'],
                    "inline": True
                })
            
            # Add suggestions count
            if 'num_suggestions' in details:
                fields.append({
                    "name": "💡 Suggestions",
                    "value": str(details['num_suggestions']),
                    "inline": True
                })
            
            # Add labels processed
            if 'num_labels' in details:
                fields.append({
                    "name": "🏷️ Labels Processed",
                    "value": str(details['num_labels']),
                    "inline": True
                })
            
            # Add confidence stats if available
            if details.get('confidence_stats'):
                stats = details['confidence_stats']
                fields.append({
                    "name": "📈 Confidence",
                    "value": f"Avg: {stats.get('avg', 0):.3f}\nMax: {stats.get('max', 0):.3f}\nMin: {stats.get('min', 0):.3f}",
                    "inline": True
                })
            
            # Add category breakdown if available
            if details.get('category_breakdown'):
                categories = details['category_breakdown']
                category_text = "\n".join([f"{cat}: {count}" for cat, count in categories.items()])
                fields.append({
                    "name": "📂 Categories",
                    "value": category_text or "None",
                    "inline": True
                })
            
            # Add cost if available
            if 'total_cost_usd' in details:
                fields.append({
                    "name": "💰 Estimated Cost",
                    "value": f"${details['total_cost_usd']:.6f}",
                    "inline": True
                })
            
            # Add tokens if available
            if 'total_tokens' in details:
                fields.append({
                    "name": "🔤 Total Tokens",
                    "value": str(details['total_tokens']),
                    "inline": True
                })
            
            # Add error message if failed
            if status == 'error' and details.get('error_message'):
                fields.append({
                    "name": "❌ Error",
                    "value": f"```{details['error_message'][:200]}```",
                    "inline": False
                })
            
            embed = {
                "title": f"📬 {endpoint} Request",
                "color": color,
                "fields": fields,
                "timestamp": details.get('timestamp'),
                "footer": {
                    "text": "Tag Suggestions API"
                }
            }
            
            payload = {
                "embeds": [embed]
            }
            
            # Send to Discord
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=5
            )
            
            if response.status_code not in (200, 204):
                logger.warning(
                    "Discord webhook failed",
                    status_code=response.status_code,
                    response=response.text[:200]
                )
            else:
                logger.debug("Discord notification sent", request_id=request_id)
                
        except Exception as e:
            # Don't fail requests because of notification errors
            logger.error(
                "Error sending Discord notification",
                error_type=type(e).__name__,
                error_message=str(e)
            )
    
    def _get_status_color(self, status: str) -> int:
        """Get Discord embed color based on status."""
        colors = {
            'success': 0x00FF00,  # Green
            'error': 0xFF0000,    # Red
            'warning': 0xFFA500   # Orange
        }
        return colors.get(status.lower(), 0x808080)  # Default gray
    
    def send_training_summary(
        self,
        num_lectures: int,
        num_prototypes: int,
        num_low_data_tags: int,
        duration_ms: float,
        status: str
    ):
        """Send training completion summary to Discord."""
        if not self.enabled:
            return
        
        try:
            color = self._get_status_color(status)
            
            embed = {
                "title": "🎓 Training Completed",
                "color": color,
                "fields": [
                    {
                        "name": "📚 Lectures",
                        "value": str(num_lectures),
                        "inline": True
                    },
                    {
                        "name": "🎯 Prototypes",
                        "value": str(num_prototypes),
                        "inline": True
                    },
                    {
                        "name": "⚠️ Low Data Tags",
                        "value": str(num_low_data_tags),
                        "inline": True
                    },
                    {
                        "name": "⏱️ Duration",
                        "value": f"{duration_ms:.2f}ms",
                        "inline": True
                    }
                ],
                "footer": {
                    "text": "Tag Suggestions API - Training"
                }
            }
            
            requests.post(self.webhook_url, json={"embeds": [embed]}, timeout=5)
            
        except Exception as e:
            logger.error(
                "Error sending training notification",
                error_type=type(e).__name__,
                error_message=str(e)
            )
