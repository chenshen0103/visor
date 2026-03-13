"""
RedFlagAnalyzer — heuristic pattern matching for common scam indicators.
Matches URLs, urgency keywords, and sensitive action phrases.
"""

import re
from dataclasses import dataclass
from typing import List, Set

@dataclass
class RedFlag:
    key: str
    description: str
    severity: float  # 0 to 1 boost to scam probability

class RedFlagAnalyzer:
    def __init__(self):
        # 1. Suspicious URL patterns (shorteners and low-cost TLDs often used in scams)
        self.url_pattern = re.compile(
            r'https?://[^\s/$.?#].[^\s]*', re.IGNORECASE
        )
        self.suspicious_tlds = {'.top', '.xyz', '.site', '.vip', '.club', '.shop', '.online', '.icu', '.app'}
        self.url_shorteners = {'bit.ly', 'tinyurl.com', 't.co', 'reurl.cc', 'ppt.cc', 'lihi.cc', 'cutt.ly'}

        # 2. Urgency and Threat keywords (Traditional Chinese)
        self.urgency_keywords = {
            "立即", "趕快", "儘速", "限時", "最後機會", "逾期", "否則", "即刻", "馬上",
            "urgent", "immediately", "asap", "last chance", "otherwise", "expired"
        }

        # 3. Sensitive Actions (Traditional Chinese)
        self.sensitive_actions = {
            "轉帳", "匯款", "提款", "點數", "遊戲點數", "ATM", "監管", "安全帳戶",
            "驗證碼", "密碼", "OTP", "身分證", "雙證件", "印章",
            "transfer", "wire", "withdraw", "points", "password", "verification code"
        }

        # 4. Contact redirection
        self.contact_patterns = [
            re.compile(r'加\s*LINE', re.IGNORECASE),
            re.compile(r'LINE\s*ID', re.IGNORECASE),
            re.compile(r'Telegram', re.IGNORECASE),
            re.compile(r'TG\s*:', re.IGNORECASE),
        ]

    def analyze(self, text: str) -> List[RedFlag]:
        flags = []
        
        # Check for suspicious URLs
        urls = self.url_pattern.findall(text)
        for url in urls:
            is_suspicious = False
            if any(shortener in url.lower() for shortener in self.url_shorteners):
                is_suspicious = True
                flags.append(RedFlag("url_shortener", f"Suspicious URL shortener detected: {url}", 0.3))
            
            if not is_suspicious and any(url.lower().endswith(tld) or (tld + "/") in url.lower() for tld in self.suspicious_tlds):
                flags.append(RedFlag("suspicious_tld", f"Low-reputation TLD detected: {url}", 0.25))

        # Check for urgency
        found_urgency = [kw for kw in self.urgency_keywords if kw in text]
        if found_urgency:
            flags.append(RedFlag("urgency", f"Urgency/Threat keywords found: {', '.join(found_urgency[:3])}", 0.15))

        # Check for sensitive actions
        found_actions = [kw for kw in self.sensitive_actions if kw in text]
        if found_actions:
            flags.append(RedFlag("sensitive_action", f"Sensitive financial actions mentioned: {', '.join(found_actions[:3])}", 0.2))

        # Check for contact redirection
        for pattern in self.contact_patterns:
            if pattern.search(text):
                flags.append(RedFlag("contact_redirect", "Attempt to move conversation to private platform (LINE/TG)", 0.25))
                break

        return flags
