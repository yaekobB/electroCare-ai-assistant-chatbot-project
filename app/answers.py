# Keep answers short, structured, and linkable.
# Use with: from answers import get_answer

ANSWERS = {
    "greeting": {
        "reply": "Hello! How can I help you today?",
        "links": [],
        "suggestions": ["View services", "Return policy", "Shipping info"]
    },
    "services_info": {
        "reply": "We offer device repairs, electronics sales, software installation, and tech support.",
        "links": [{"title": "Our Services", "url": "/services"}],
        "suggestions": ["How to order", "Contact support"]
    },
    "website_navigation": {
        "reply": "Tell me what you’re looking for and I’ll open the right page.",
        "links": [
            {"title": "Help Center", "url": "/help/faq"},
            {"title": "Account", "url": "/account"}
        ],
        "suggestions": ["Open Return Policy", "Open Shipping Info", "Go to Account Settings"]
    },
    "payment_help": {
        "reply": "You can pay with cards, PayPal, Apple Pay, and bank transfer. Payments are encrypted.",
        "links": [{"title": "Payment Methods", "url": "/help/payments"}],
        "suggestions": ["How to order", "Contact support"]
    },
    "return_policy_info": {
        "reply": "Items can be returned within 30 days in original condition. Refunds are issued after inspection.",
        "links": [{"title": "Return Policy", "url": "/help/returns"}],
        "suggestions": ["Refund status", "Shipping info"]
    },
    "refund_status": {
        "reply": "Refunds are typically processed within 3–5 business days after the return is inspected.",
        "links": [{"title": "Refunds (Return Policy)", "url": "/help/returns#refunds"}],
        "suggestions": ["Return policy", "Contact support"]
    },
    "contact_support": {
        "reply": "You can reach a human via live chat or the contact form.",
        "links": [
            {"title": "Live Chat", "url": "/support/chat"},
            {"title": "Contact Form", "url": "/support/contact"}
        ],
        "suggestions": ["View services", "Return policy"]
    },
    "promotion_info": {
        "reply": "We run seasonal promotions and occasional coupon codes.",
        "links": [{"title": "Pricing & Offers", "url": "/pricing"}],
        "suggestions": ["Student discount", "Payment methods"]
    },
    "end_chat": {
        "reply": "Thanks for chatting! If you need anything else, I’m here.",
        "links": [],
        "suggestions": []
    },
    "faq_shipping": {
        "reply": "We offer standard and express delivery. You’ll get a tracking link after dispatch.",
        "links": [{"title": "Shipping Information", "url": "/help/shipping"}],
        "suggestions": ["Return policy", "Contact support"]
    },
    "faq_account": {
        "reply": "Manage your account settings, password, and addresses from your Account page.",
        "links": [
            {"title": "Account", "url": "/account"},
            {"title": "Reset Password", "url": "/account/reset-password"}
        ],
        "suggestions": ["How to order", "Contact support"]
    },
    "faq_warranty": {
        "reply": "Most products include a limited warranty. Terms vary by product.",
        "links": [{"title": "Warranty", "url": "/help/warranty"}],
        "suggestions": ["Return policy", "Contact support"]
    },
    "faq_store_info": {
        "reply": "You can visit our stores during posted hours. Some services offer in‑store returns and pickup.",
        "links": [{"title": "Stores & Hours", "url": "/stores"}],
        "suggestions": ["Shipping info", "Return policy"]
    },
    "faq_product_condition": {
        "reply": "We sell new items and clearly mark refurbished products when applicable.",
        "links": [{"title": "Help Center", "url": "/help/faq"}],
        "suggestions": ["Product quality", "Warranty"]
    },
    "faq_product_quality": {
        "reply": "We work with trusted brands and perform quality checks on refurbished items.",
        "links": [{"title": "Help Center", "url": "/help/faq"}],
        "suggestions": ["Warranty", "Tech specs"]
    },
    "faq_tech_specs": {
        "reply": "Detailed specifications are listed on each product page.",
        "links": [{"title": "All Products", "url": "/products"}],
        "suggestions": ["Laptops", "Phones"]
    },
    "faq_referral_program": {
        "reply": "Invite friends and earn credits when they make a purchase.",
        "links": [{"title": "Referral Program", "url": "/help/referrals"}],
        "suggestions": ["Pricing & Offers", "Subscription"]
    },
    "faq_security_privacy": {
        "reply": "We encrypt payments and protect your data according to our privacy policy.",
        "links": [
            {"title": "Security", "url": "/legal/security"},
            {"title": "Privacy Policy", "url": "/legal/privacy"}
        ],
        "suggestions": ["Payment methods", "Contact support"]
    },
    "faq_delivery_problems": {
        "reply": "If your package is missing or damaged, contact support with your order details.",
        "links": [
            {"title": "Shipping Help", "url": "/help/shipping"},
            {"title": "Contact Support", "url": "/support/contact"}
        ],
        "suggestions": ["Return policy", "Live chat"]
    },
    "faq_return_timeframe": {
        "reply": "The standard return window is 30 days from delivery (unless otherwise noted).",
        "links": [{"title": "Return Policy", "url": "/help/returns"}],
        "suggestions": ["Refund status", "Contact support"]
    },
    "faq_exchange_policy": {
        "reply": "Exchanges may be available depending on stock and condition.",
        "links": [{"title": "Return & Exchange Policy", "url": "/help/returns"}],
        "suggestions": ["Shipping info", "Contact support"]
    },
    "faq_payment_methods": {
        "reply": "We accept major cards, PayPal, Apple Pay, and bank transfers.",
        "links": [{"title": "Payment Methods", "url": "/help/payments"}],
        "suggestions": ["How to order", "Security & privacy"]
    },
    "faq_student_discount": {
        "reply": "Students may qualify for special pricing after verification.",
        "links": [{"title": "Student Discount", "url": "/help/student-discount"}],
        "suggestions": ["Pricing & Offers", "Subscription"]
    },
    "faq_order_confirmation": {
        "reply": "You’ll receive an email confirmation shortly after placing an order.",
        "links": [{"title": "Order Confirmation Help", "url": "/help/order-confirmation"}],
        "suggestions": ["Payment methods", "Contact support"]
    },
    "faq_bulk_purchase": {
        "reply": "We support bulk orders and can offer tiered pricing for businesses.",
        "links": [{"title": "Bulk Orders", "url": "/business/bulk"}],
        "suggestions": ["Contact support", "Pricing & Offers"]
    },
    "faq_subscription": {
        "reply": "Subscribe for product updates and promotions by email.",
        "links": [{"title": "Newsletter", "url": "/subscribe"}],
        "suggestions": ["Student discount", "Pricing & Offers"]
    },
    "faq_chatbot_capability": {
        "reply": "I’m an info assistant: I can explain services, steps, and where to find things on the site. I can’t access or change orders.",
        "links": [{"title": "Help Center", "url": "/help/faq"}],
        "suggestions": ["View services", "Return policy", "Shipping info"]
    },
    "faq_price_negotiation": {
        "reply": "We occasionally run promotions and limited‑time offers. Price matching may apply to selected products.",
        "links": [{"title": "Pricing & Offers", "url": "/pricing"}],
        "suggestions": ["Student discount", "Referral program"]
    }
}

def get_answer(intent: str):
    # Default safe response
    return ANSWERS.get(
        intent,
        {
            "reply": "I can help with that. Do you want shipping info, return policy, or payment methods?",
            "links": [
                {"title": "Help Center", "url": "/help/faq"}
            ],
            "suggestions": ["Shipping info", "Return policy", "Payment methods"]
        }
    )
