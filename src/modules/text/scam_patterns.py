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
    ScamArchetype(
        key="guess_who_i_am",
        name_en="Guess Who I Am (Friend/Relative Impersonation)",
        name_zh="猜猜我是誰（冒充親友）",
        description=(
            "Fraudster calls or messages pretending to be a friend or relative who "
            "has changed their number. They build rapport before asking for urgent "
            "financial help due to an emergency or business trouble."
        ),
        exemplars=[
            "猜猜我是誰？我是你大學同學啦，我換電話了，先存一下",
            "大舅你聽得出來我是誰嗎？我最近急需用錢，能不能先借我10萬",
            "Hi, it's me, your old friend. I lost my phone and this is my new number.",
            "I'm in a bit of a jam and need some quick cash, can you help me out?",
            "我是你侄子，我現在人在醫院急需繳費，能不能先匯款給我",
            "好久不見！我換LINE了，這是我的新帳號，有空出來吃飯嗎",
            "Can you wire me some money? I'm stranded and my wallet was stolen.",
            "阿姨是我啦，我最近在創業需要週轉，能不能借我一點錢",
        ],
    ),
    ScamArchetype(
        key="atm_deduction_fraud",
        name_en="ATM / Deduction Error Fraud",
        name_zh="解除分期付款詐騙",
        description=(
            "Fraudster poses as e-commerce or bank staff claiming a system error "
            "caused an incorrect recurring charge or membership upgrade. They "
            "instruct the victim to go to an ATM or use net banking to 'cancel' it."
        ),
        exemplars=[
            "您好，我是博客來客服，因系統設定錯誤，您的訂單被重複扣款",
            "您的訂單被誤設為批發商，請至ATM操作解除分期付款設定",
            "There was an error with your subscription, please follow these steps to refund",
            "Your payment was processed twice, go to an ATM to reverse the transaction",
            "我是網路賣場客服，因工作人員失誤將您設為VIP，需操作網銀解除",
            "您的信用卡將被自動扣款12期，請配合客服人員取消此筆交易",
            "System error: your account will be charged monthly, call this number to cancel",
            "因訂單系統異常，請您到最近的ATM依照指示解除設定",
        ],
    ),
    ScamArchetype(
        key="job_scam",
        name_en="Job / Employment Scam",
        name_zh="求職／家庭代工詐騙",
        description=(
            "Promises of high-paying, low-effort jobs or work-from-home "
            "opportunities. Victims are asked to pay 'setup fees', provide bank "
            "account details for 'payroll' (which are used for money laundering), "
            "or complete tasks on fake platforms."
        ),
        exemplars=[
            "在家工作，每日只需1小時，月入5萬不是夢，意者加LINE",
            "誠徵打字員、點讚員，每單佣金300元，現結不拖欠",
            "Work from home and earn $500 a day, no experience required",
            "High paying part-time job, just need your bank account to receive payments",
            "急徵家庭代工，需先繳納材料保證金2000元，完工後退還",
            "電商平台刷單員，幫賣場刷好評即可獲得高額提成",
            "Easy money: process payments for our international clients from home",
            "招聘線上客服，只需手機即可操作，薪資優渥環境自由",
        ],
    ),
    ScamArchetype(
        key="phishing_link_fraud",
        name_en="Phishing / Identity Theft",
        name_zh="釣魚連結／個資盜取",
        description=(
            "SMS or emails containing fake links to banks, government portals, or "
            "logistics companies. Victims are lulled into entering credentials, "
            "credit card numbers, or OTPs on a fraudulent website."
        ),
        exemplars=[
            "您的網銀帳戶異常，請立即登入更新資料，否則將永久凍結",
            "【國稅局通知】您有一筆退稅尚未領取，請點擊連結填寫匯款帳號",
            "Your bank account security is compromised, click here to verify identity",
            "You have an unclaimed reward, login now to claim your $500 gift card",
            "您的電費已逾期，請立即點擊連結繳費，以免造成斷電",
            "【蝦皮購物】您的帳號異地登入，若非本人請點此登入修改密碼",
            "Verify your Netflix account info to avoid service interruption",
            "您的交通罰單未繳納，請點擊連結查詢詳細內容並完成繳費",
        ],
    ),
]

# Build a quick lookup dict
ARCHETYPE_BY_KEY = {a.key: a for a in SCAM_ARCHETYPES}
