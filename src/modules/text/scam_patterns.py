"""
scam_patterns — hard-coded scam archetype definitions for intent matching.

Each archetype contains:
- key          : internal ID
- name_en      : English label
- name_zh      : Chinese label
- description  : Free-text description (used for embedding)
- exemplars    : Representative scam phrases (bilingual, used for centroid embedding)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class ScamArchetype:
    key: str
    name_en: str
    name_zh: str
    description: str
    exemplars: List[str]


SCAM_ARCHETYPES: List[ScamArchetype] = [
    ScamArchetype(
        key="investment_fraud",
        name_en="Investment Fraud",
        name_zh="高報酬投資詐騙",
        description=(
            "Fraudulent schemes promising unusually high or guaranteed investment "
            "returns, often involving fake trading platforms, cryptocurrency scams, "
            "or Ponzi structures.  The victim is lured with small 'proof' profits "
            "before being asked to invest large sums."
        ),
        exemplars=[
            "我們的平台保證每月30%回報，零風險投資機會",
            "加入我們的加密貨幣群組，已有成員月賺百萬",
            "This trading bot guarantees 50% profit every month, join now",
            "Guaranteed returns on forex investment, no risk at all",
            "限時優惠！投資10萬，三個月翻三倍，名額有限",
            "Our AI trading system has 98% win rate, minimum investment NT$50,000",
            "朋友介紹的股票內線消息，明天漲停，趕快買進",
            "Virtual currency arbitrage opportunity, wire $10,000 to get started",
        ],
    ),
    ScamArchetype(
        key="romance_fraud",
        name_en="Romance / Pig-butchering Fraud",
        name_zh="假交友／殺豬盤詐騙",
        description=(
            "Fraudster builds a romantic or friendly relationship online over weeks "
            "or months before introducing a 'life-changing' investment or asking for "
            "urgent financial help.  Common on dating apps, social media, and LINE."
        ),
        exemplars=[
            "我在新加坡工作，賺了很多錢，想和你分享投資秘訣",
            "我很喜歡你，我有個朋友教我的投資方法，你要不要試試",
            "I'm overseas right now and need you to transfer money for an emergency",
            "We've been talking for weeks, I trust you, please help me financially",
            "我是外籍工程師，在海外工作，想介紹你認識好的投資平台",
            "我們的感情那麼好，你能借我一些錢嗎？我回國馬上還你",
            "My overseas account is frozen, please transfer USD 5,000 to help me",
            "認識你這麼久，你是我最信任的人，這個投資機會只給你",
        ],
    ),
    ScamArchetype(
        key="government_impersonation",
        name_en="Government / Authority Impersonation",
        name_zh="假冒公務機關詐騙",
        description=(
            "Fraudsters pose as police, prosecutors, courts, tax authorities, social "
            "security, or other government bodies.  The victim is told their account "
            "or identity is implicated in a crime and must transfer funds to a "
            "'supervised account' to prove innocence."
        ),
        exemplars=[
            "您好，刑事局通知您帳戶涉嫌洗錢，請配合轉帳至監管帳戶",
            "這裡是台北地檢署，您的帳戶被犯罪集團盜用，需要配合調查",
            "This is the IRS calling, you owe back taxes and face immediate arrest",
            "Hello, this is Social Security Administration, your number was suspended",
            "您的健保資料遭盜用，請立即撥打我們的專線，否則將凍結您的帳戶",
            "法院通知您有未出庭的案件，請轉帳保證金否則將發出逮捕令",
            "警察局來電：您的帳戶被列為詐騙共犯，配合轉帳以示清白",
            "Your package contains illegal items; please cooperate with our officers",
        ],
    ),
    ScamArchetype(
        key="parcel_fraud",
        name_en="Parcel / Logistics Fraud",
        name_zh="包裹／物流詐騙",
        description=(
            "Victim receives a message claiming a parcel cannot be delivered, is "
            "held at customs, or has been incorrectly addressed.  They are asked to "
            "click a link, pay a fee, or provide personal / banking information to "
            "release the package."
        ),
        exemplars=[
            "您的包裹因地址不完整無法投遞，請點擊連結更新資料",
            "您有一個包裹在海關滯留，請支付關稅NT$850以便放行",
            "Your DHL package requires customs clearance, pay $25 to release",
            "Attention: your parcel delivery failed, click here to reschedule",
            "黑貓宅急便通知：您的貨物需補繳運費，請在24小時內完成付款",
            "您的Amazon訂單無法配送，請更新您的付款資訊",
            "FedEx alert: package held, click link to verify your address immediately",
            "您訂購的商品已到達，請支付補充運費，否則將退回給寄件人",
        ],
    ),
]

# Build a quick lookup dict
ARCHETYPE_BY_KEY = {a.key: a for a in SCAM_ARCHETYPES}
