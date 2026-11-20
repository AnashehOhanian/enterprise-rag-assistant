"""Client and shop language text for the Safeguard 4U chat.

The client can use English, Armenian, Farsi, or Spanish. The shop review
uses English or Farsi. Dollar ranges stay numeric. No rate tables live here.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote

CLIENT_LANGUAGES = ("en", "hy", "fa", "es")
SHOP_LANGUAGES = ("en", "fa")
LANGUAGE_LABELS = {
    "en": "English",
    "hy": "Հայերեն",
    "fa": "فارسی",
    "es": "Español",
}

GATE_WORDS = ("gate", "gates", "դարպաս", "دروازه", "puerta", "portón", "porton")
PATIO_WORDS = ("patio", "pergola", "outdoor", "պատշգամբ", "պերգոլա", "پاسیو", "آلاچیق", "pérgola")
STAIR_WORDS = ("stair", "stairs", "staircase", "աստիճան", "پله", "escalera", "escaleras")
RAILING_WORDS = (
    "railing",
    "railings",
    "balustrade",
    "handrail",
    "բազրիք",
    "نرده",
    "baranda",
    "barandal",
    "barandilla",
)
MODERN_WORDS = (
    "modern",
    "horizontal",
    "contemporary",
    "ժամանակակից",
    "մոդեռն",
    "հորիզոնական",
    "مدرن",
    "افقی",
    "moderno",
    "contemporáneo",
    "contemporaneo",
)
ORNAMENTAL_WORDS = (
    "ornamental",
    "classic",
    "traditional",
    "դեկորատիվ",
    "դասական",
    "فرفورژه",
    "کلاسیک",
    "تزئینی",
    "clásico",
    "clasico",
    "tradicional",
)
DOUBLE_WORDS = ("double", "կրկնակի", "دو لنگه", "doble")
EXAMPLE_STAIR_WORDS = STAIR_WORDS + RAILING_WORDS
SMALL_GATE_PHRASES = (
    "small",
    "backyard",
    "back yard",
    "walk gate",
    "pedestrian",
    "side gate",
    "yard gate",
    "փոքր",
    "բակի դարպաս",
    "کوچک",
    "pequeño",
    "pequeña",
    "jardin",
    "jardín",
)
INCH_WORDS = ("inch", "inches", "դյույմ", "اینچ", "pulgada", "pulgadas")
FOOT_WORDS = ("foot", "feet", "ft", "ֆուտ", "فوت", "pie", "pies")
WHITE_WORDS = ("white", "blanco", "blanca", "սպիտակ", "سفید")
TIMING_PHRASES = (
    "how long",
    "timeline",
    "lead time",
    "when can",
    "how soon",
    "որքան ժամանակ",
    "ժամկետ",
    "چقدر طول",
    "زمان نصب",
    "cuánto tarda",
    "cuanto tarda",
    "plazo",
)
TIMING_MARKERS = ("workload", "ծանրաբեռնվածություն", "حجم کار", "carga de trabajo")
FINISH_MARKERS = ("finish", "հարդարում", "رنگ نهایی", "acabado")
WHICH_MARKERS = ("which one", "որ մեկը", "کدام", "cuál", "cual")
SAMPLE_MARKERS = (
    "sample",
    "photo",
    "picture",
    "նմուշ",
    "լուսանկար",
    "نمونه",
    "عکس",
    "muestra",
    "foto",
)
EXAMPLE_CUES = (
    "նմուշ",
    "օրինակ",
    "լուսանկար",
    "նկար",
    "ցույց տուր",
    "ցույց տվեք",
    "نمونه",
    "مثال",
    "عکس",
    "تصویر",
    "نشان بده",
    "ejemplo",
    "muestra",
    "foto",
    "imagen",
    "muéstrame",
    "muestrame",
)
ALL_EXAMPLE_PHRASES = (
    "բոլոր ոճերը",
    "همه سبک",
    "todos los estilos",
    "todos los ejemplos",
)
SIZE_QUESTION_PHRASES = (
    "ինչ չափ",
    "چه اندازه",
    "qué medida",
    "que medida",
    "ancho",
    "alto",
)

PHRASES: dict[str, dict[str, str]] = {
    "language": {
        "en": "Language",
        "hy": "Լեզու",
        "fa": "زبان",
        "es": "Idioma",
    },
    "shop_language": {
        "en": "Shop language",
        "fa": "زبان کارگاه",
    },
    "ask_title": {
        "en": "Ask about ornamental steel",
        "hy": "Հարցրեք դեկորատիվ պողպատի մասին",
        "fa": "درباره فولاد تزئینی بپرسید",
        "es": "Pregunte sobre acero ornamental",
    },
    "banner": {
        "en": "Ranges are rough until the shop checks the site. Timing depends on workload, measurements, and scope.",
        "hy": "Միջակայքերը նախնական են, մինչև արհեստանոցը ստուգի տեղանքը։ Ժամկետը կախված է ծանրաբեռնվածությունից, չափումներից և աշխատանքի ծավալից։",
        "fa": "بازه‌ها تقریبی هستند تا کارگاه محل را بررسی کند. زمان‌بندی به حجم کار، اندازه‌گیری و محدوده کار بستگی دارد.",
        "es": "Los rangos son aproximados hasta que el taller revise el sitio. El plazo depende de la carga de trabajo, las medidas y el alcance.",
    },
    "photo_label": {
        "en": "Photo of the yard, opening, stairs, or patio",
        "hy": "Բակի, բացվածքի, աստիճանների կամ պատշգամբի լուսանկար",
        "fa": "عکس حیاط، دهانه، پله یا پاسیو",
        "es": "Foto del jardín, el vano, la escalera o el patio",
    },
    "photo_unread": {
        "en": "That photo could not be read. Use a PNG, JPEG, or WebP.",
        "hy": "Այդ լուսանկարը չհաջողվեց կարդալ։ Օգտագործեք PNG, JPEG կամ WebP։",
        "fa": "این عکس خوانده نشد. از PNG، JPEG یا WebP استفاده کنید.",
        "es": "No se pudo leer esa foto. Use PNG, JPEG o WebP.",
    },
    "photo_help": {
        "en": "The preview stays on this page. Send to the shop includes it with your questions.",
        "hy": "Նախադիտումը մնում է այս էջում։ Արհեստանոց ուղարկելիս այն գալիս է ձեր հարցերի հետ։",
        "fa": "پیش‌نمایش در همین صفحه می‌ماند. با فرستادن به کارگاه، همراه پرسش‌ها می‌رود.",
        "es": "La vista previa se queda en esta página. Al enviar al taller, va con sus preguntas.",
    },
    "earlier": {
        "en": "Show earlier messages ({count})",
        "hy": "Ցույց տալ ավելի վաղ հաղորդագրությունները ({count})",
        "fa": "نمایش پیام‌های قبلی ({count})",
        "es": "Mostrar mensajes anteriores ({count})",
    },
    "turn_limit": {
        "en": "This conversation has 8 questions. Send them to the shop, or start a new one.",
        "hy": "Այս զրույցում արդեն 8 հարց կա։ Ուղարկեք դրանք արհեստանոց, կամ սկսեք նորը։",
        "fa": "این گفتگو ۸ پرسش دارد. آن‌ها را به کارگاه بفرستید یا گفتگوی تازه شروع کنید.",
        "es": "Esta conversación ya tiene 8 preguntas. Envíelas al taller o empiece una nueva.",
    },
    "chat_placeholder": {
        "en": "Ask about a railing, gate, stair, or patio",
        "hy": "Հարցրեք բազրիքի, դարպասի, աստիճանի կամ պատշգամբի մասին",
        "fa": "درباره نرده، دروازه، پله یا پاسیو بپرسید",
        "es": "Pregunte por una baranda, un portón, una escalera o un patio",
    },
    "new_conversation": {
        "en": "Start a new conversation",
        "hy": "Սկսել նոր զրույց",
        "fa": "شروع گفتگوی تازه",
        "es": "Empezar una conversación nueva",
    },
    "send_title": {
        "en": "Send questions to the shop",
        "hy": "Հարցերն ուղարկել արհեստանոց",
        "fa": "فرستادن پرسش‌ها به کارگاه",
        "es": "Enviar preguntas al taller",
    },
    "send_caption": {
        "en": "Add a note or a photo. If you have chatted, you can include those questions and the rough estimate.",
        "hy": "Ավելացրեք գրառում կամ լուսանկար։ Եթե զրուցել եք, կարող եք ներառել այդ հարցերն ու նախնական գնահատականը։",
        "fa": "یک یادداشت یا عکس اضافه کنید. اگر گفتگو کرده‌اید، می‌توانید آن پرسش‌ها و برآورد تقریبی را هم بفرستید.",
        "es": "Añada una nota o una foto. Si ya conversó, puede incluir esas preguntas y el estimado aproximado.",
    },
    "include_estimate": {
        "en": "Include the rough estimate",
        "hy": "Ներառել նախնական գնահատականը",
        "fa": "شامل برآورد تقریبی",
        "es": "Incluir el estimado aproximado",
    },
    "need_shop_detail": {
        "en": "Add a note, a photo, or a chat question before sending.",
        "hy": "Ուղարկելուց առաջ ավելացրեք գրառում, լուսանկար կամ զրույցի հարց։",
        "fa": "پیش از فرستادن، یک یادداشت، عکس، یا پرسش گفتگو اضافه کنید.",
        "es": "Añada una nota, una foto o una pregunta del chat antes de enviar.",
    },
    "ask_first": {
        "en": "Ask a question first, then choose what the shop should see.",
        "hy": "Նախ հարց տվեք, հետո ընտրեք, թե ինչը տեսնի արհեստանոցը։",
        "fa": "اول یک پرسش بنویسید، بعد انتخاب کنید کارگاه چه چیزی را ببیند.",
        "es": "Haga una pregunta primero y luego elija lo que el taller debe ver.",
    },
    "questions_include": {
        "en": "Questions to include",
        "hy": "Ներառվող հարցեր",
        "fa": "پرسش‌هایی که فرستاده شوند",
        "es": "Preguntas para incluir",
    },
    "name": {"en": "Name", "hy": "Անուն", "fa": "نام", "es": "Nombre"},
    "contact": {
        "en": "Phone or email",
        "hy": "Հեռախոս կամ էլ․ փոստ",
        "fa": "تلفن یا ایمیل",
        "es": "Teléfono o correo",
    },
    "note_for_shop": {
        "en": "Note for the shop",
        "hy": "Գրառում արհեստանոցի համար",
        "fa": "یادداشت برای کارگاه",
        "es": "Nota para el taller",
    },
    "send_button": {
        "en": "Send to the shop",
        "hy": "Ուղարկել արհեստանոց",
        "fa": "فرستادن به کارگاه",
        "es": "Enviar al taller",
    },
    "sent": {
        "en": "Sent to Safeguard 4U shop.",
        "hy": "Ուղարկված է Safeguard 4U արհեստանոց։",
        "fa": "به کارگاه Safeguard 4U فرستاده شد.",
        "es": "Enviado al taller de Safeguard 4U.",
    },
    "sms_via_twilio": {
        "en": "Text sent through Twilio.",
        "hy": "Հաղորդագրությունն ուղարկվել է Twilio-ով։",
        "fa": "پیامک از طریق Twilio فرستاده شد.",
        "es": "Texto enviado por Twilio.",
    },
    "sms_via_email": {
        "en": "Text sent through email-to-SMS. Twilio is not configured.",
        "hy": "Հաղորդագրությունն ուղարկվել է email-to-SMS-ով։ Twilio-ն կարգավորված չէ։",
        "fa": "پیامک از طریق ایمیل‌به‌پیامک فرستاده شد. Twilio تنظیم نشده است.",
        "es": "Texto enviado por correo-a-SMS. Twilio no está configurado.",
    },
    "sms_via_fallback": {
        "en": "Twilio did not send the text. It went through email-to-SMS instead.",
        "hy": "Twilio-ն չուղարկեց հաղորդագրությունը։ Այն գնաց email-to-SMS-ով։",
        "fa": "Twilio پیامک را نفرستاد. به جایش از ایمیل‌به‌پیامک رفت.",
        "es": "Twilio no envió el texto. Se envió por correo-a-SMS.",
    },
    "sms_skipped": {
        "en": "Text was not sent. Twilio and email-to-SMS settings in .env are not filled in yet. The request was saved.",
        "hy": "Հաղորդագրությունը չուղարկվեց։ Twilio-ն և email-to-SMS կարգավորումները .env-ում դեռ լրացված չեն։ Հարցումը պահվել է։",
        "fa": "پیامک فرستاده نشد. تنظیمات Twilio و ایمیل‌به‌پیامک در .env هنوز پر نشده است. درخواست ذخیره شد.",
        "es": "El texto no se envió. Twilio y el correo-a-SMS todavía no están completos en .env. La solicitud quedó guardada.",
    },
    "sms_failed": {
        "en": "Text was not sent. The gateway did not accept it. The request was saved.",
        "hy": "Հաղորդագրությունը չուղարկվեց։ Դարպասը այն չընդունեց։ Հարցումը պահվել է։",
        "fa": "پیامک فرستاده نشد. درگاه آن را نپذیرفت. درخواست ذخیره شد.",
        "es": "El texto no se envió. La pasarela no lo aceptó. La solicitud quedó guardada.",
    },
    "email_sent": {
        "en": "Full details were emailed to the shop.",
        "hy": "Ամբողջական մանրամասներն ուղարկվել են արհեստանոցի էլ․ փոստին։",
        "fa": "جزئیات کامل به ایمیل کارگاه فرستاده شد.",
        "es": "Los detalles completos se enviaron al correo del taller.",
    },
    "email_failed": {
        "en": "The shop email was not sent. The request was saved.",
        "hy": "Նամակը չուղարկվեց։ Հարցումը պահվել է։",
        "fa": "ایمیل کارگاه فرستاده نشد. درخواست ذخیره شد.",
        "es": "El correo del taller no se envió. La solicitud quedó guardada.",
    },
    "no_answer": {
        "en": "The shop assistant did not return an answer.",
        "hy": "Արհեստանոցի օգնականը պատասխան չվերադարձրեց։",
        "fa": "دستیار کارگاه پاسخی برنگرداند.",
        "es": "El asistente del taller no devolvió una respuesta.",
    },
    "rate_limit": {
        "en": "Please wait about {seconds} seconds before sending more.",
        "hy": "Խնդրում ենք սպասել մոտ {seconds} վայրկյան, հետո նորից ուղարկել։",
        "fa": "لطفاً حدود {seconds} ثانیه صبر کنید و دوباره بفرستید.",
        "es": "Espere unos {seconds} segundos antes de enviar más.",
    },
    "shop_requests": {"en": "Shop inbox", "fa": "صندوق کارگاه"},
    "no_requests": {"en": "No requests yet.", "fa": "هنوز درخواستی نیست."},
    "no_name": {"en": "No name", "fa": "بدون نام"},
    "no_contact": {"en": "No contact", "fa": "بدون تماس"},
    "customer_questions": {
        "en": "Customer questions",
        "fa": "پرسش‌های مشتری",
    },
    "site_photo": {"en": "Site photo", "fa": "عکس محل"},
    "extra_note": {"en": "Customer note", "fa": "یادداشت مشتری"},
    "alert_heading": {
        "en": "Safeguard 4U request {reference}",
        "fa": "درخواست Safeguard 4U {reference}",
    },
    "alert_customer": {"en": "Customer: {name}", "fa": "مشتری: {name}"},
    "alert_contact": {"en": "Customer contact: {contact}", "fa": "تماس مشتری: {contact}"},
    "copy_summary": {"en": "Text summary", "fa": "خلاصه پیام"},
    "send_text": {
        "en": "Send text message to {phone}",
        "fa": "فرستادن پیامک به {phone}",
    },
    "text_ready": {
        "en": "Copy the summary, or open it in your text app.",
        "fa": "خلاصه را کپی کنید، یا آن را در برنامه پیامک باز کنید.",
    },
    "schedule_technician": {
        "en": "To schedule a technician, contact Safeguard 4U at {phone} or {email}.",
        "hy": "Տեխնիկի այցը նշանակելու համար կապվեք Safeguard 4U-ի հետ՝ {phone} կամ {email}։",
        "fa": "برای هماهنگی بازدید تکنسین، با Safeguard 4U به شماره {phone} یا {email} تماس بگیرید.",
        "es": "Para programar un técnico, contacte a Safeguard 4U al {phone} o en {email}.",
    },
    "shop_summary_title": {"en": "Shop summary", "fa": "خلاصه برای کارگاه"},
    "customer_language": {
        "en": "Customer language: {language}",
        "fa": "زبان مشتری: {language}",
    },
    "project_line": {"en": "Project: {project}", "fa": "پروژه: {project}"},
    "project_gate": {"en": "Gate", "hy": "Դարպաս", "fa": "دروازه", "es": "Portón"},
    "project_stair": {"en": "Stair", "hy": "Աստիճան", "fa": "پله", "es": "Escalera"},
    "project_patio": {"en": "Patio frame", "hy": "Պատշգամբի շրջանակ", "fa": "قاب پاسیو", "es": "Marco de patio"},
    "project_railing": {"en": "Railing", "hy": "Բազրիք", "fa": "نرده", "es": "Baranda"},
    "help_fallback": {
        "en": "I can help with ornamental steel, gates, stairs, railings, and patio frames.",
        "hy": "Կարող եմ օգնել դեկորատիվ պողպատի, դարպասների, աստիճանների, բազրիքների և պատշգամբի շրջանակների հարցում։",
        "fa": "می‌توانم درباره فولاد تزئینی، دروازه، پله، نرده و قاب پاسیو کمک کنم.",
        "es": "Puedo ayudar con acero ornamental, portones, escaleras, barandas y marcos de patio.",
    },
    "refusal": {
        "en": "I can help with catalog products and a rough ballpark. I can't share shop instructions.",
        "hy": "Կարող եմ օգնել կատալոգի ապրանքների և նախնական միջակայքի հարցում։ Արհեստանոցի ներքին հրահանգները չեմ կիսում։",
        "fa": "می‌توانم درباره محصولات کاتالوگ و یک برآورد تقریبی کمک کنم. دستورهای داخلی کارگاه را نمی‌گویم.",
        "es": "Puedo ayudar con los productos del catálogo y un estimado aproximado. No comparto las instrucciones internas del taller.",
    },
    "modern_look": {
        "en": "A modern look works on that opening. ",
        "hy": "Ժամանակակից տեսքը համապատասխանում է այդ բացվածքին։ ",
        "fa": "ظاهر مدرن برای این دهانه مناسب است. ",
        "es": "Un estilo moderno funciona en ese vano. ",
    },
    "unit_question": {
        "en": "{look}For a backyard gate, {width} by {height} usually means inches. Is it {width} in wide by {height} in high, or closer to {width} in wide by 72 in tall?",
        "hy": "{look}Բակի դարպասի համար {width}-ը {height}-ի հետ սովորաբար դյույմ է։ Դա {width} դյույմ լայնք և {height} դյույմ բարձրություն է, թե՞ ավելի մոտ է {width} դյույմ լայնքին և 72 դյույմ բարձրությանը։",
        "fa": "{look}برای دروازه حیاط، {width} در {height} معمولاً اینچ است. آیا {width} اینچ عرض و {height} اینچ ارتفاع است، یا نزدیک‌تر به {width} اینچ عرض و 72 اینچ ارتفاع؟",
        "es": "{look}Para un portón de jardín, {width} por {height} suele ser en pulgadas. ¿Son {width} pulgadas de ancho por {height} de alto, o está más cerca de {width} pulgadas de ancho por 72 de alto?",
    },
    "timing": {
        "en": "Exact timing depends on shop workload, the site measurements, and any permits or extra scope. A straightforward shop build is often a few weeks after the measurements are confirmed.",
        "hy": "Ճշգրիտ ժամկետը կախված է արհեստանոցի ծանրաբեռնվածությունից, տեղանքի չափումներից և թույլտվություններից կամ լրացուցիչ աշխատանքից։ Սովորական պատվերը հաճախ պատրաստ է մի քանի շաբաթում, չափումները հաստատելուց հետո։",
        "fa": "زمان دقیق به حجم کار کارگاه، اندازه‌گیری محل، و مجوز یا کار اضافه بستگی دارد. یک ساخت ساده معمولاً چند هفته پس از قطعی شدن اندازه‌ها آماده می‌شود.",
        "es": "El plazo exacto depende de la carga de trabajo del taller, las medidas del sitio y los permisos o el alcance extra. Un trabajo sencillo suele tardar unas semanas después de confirmar las medidas.",
    },
    "which_sample": {
        "en": "I can show a modern gate, an ornamental gate, a stair railing, or a patio frame. Which one should I pull up?",
        "hy": "Կարող եմ ցույց տալ ժամանակակից դարպաս, դեկորատիվ դարպաս, աստիճանի բազրիք կամ պատշգամբի շրջանակ։ Ո՞ր մեկը բացեմ։",
        "fa": "می‌توانم دروازه مدرن، دروازه فرفورژه، نرده پله یا قاب پاسیو را نشان بدهم. کدام را بیاورم؟",
        "es": "Puedo mostrar un portón moderno, un portón ornamental, una baranda de escalera o un marco de patio. ¿Cuál quiere ver?",
    },
    "sample_one": {
        "en": "Here's a sample of that look.",
        "hy": "Ահա այդ տեսքի մի նմուշ։",
        "fa": "این یک نمونه از همان طرح است.",
        "es": "Aquí hay una muestra de ese estilo.",
    },
    "samples_many": {
        "en": "Here are samples of those looks.",
        "hy": "Ահա այդ տեսքերի նմուշները։",
        "fa": "این نمونه‌هایی از همان طرح‌هاست.",
        "es": "Aquí hay muestras de esos estilos.",
    },
    "fits_opening": {
        "en": "That look fits the opening you described.",
        "hy": "Այդ տեսքը համապատասխանում է ձեր նշած բացվածքին։",
        "fa": "این طرح با دهانه‌ای که گفتید جور است.",
        "es": "Ese estilo encaja con el vano que describió.",
    },
    "need_size": {
        "en": "Width and height are enough for a rough gate range.",
        "hy": "Լայնքն ու բարձրությունը բավարար են դարպասի նախնական միջակայքի համար։",
        "fa": "عرض و ارتفاع برای یک بازه تقریبی دروازه کافی است.",
        "es": "El ancho y el alto bastan para un rango aproximado del portón.",
    },
    "usual_height": {
        "en": " The height used is the usual height for that style.",
        "hy": " Բարձրությունը այդ ոճի սովորական բարձրությունն է։",
        "fa": " ارتفاع به‌کاررفته ارتفاع معمول این سبک است.",
        "es": " La altura usada es la habitual para ese estilo.",
    },
    "white_finish": {
        "en": " White is a finish and stays on its own line.",
        "hy": " Սպիտակը հարդարում է և մնում է առանձին տողում։",
        "fa": " سفید یک رنگ نهایی است و در ردیف جداگانه می‌ماند.",
        "es": " El blanco es un acabado y queda en su propia línea.",
    },
    "ballpark": {
        "en": "A rough estimate before a site check and shop review is {range}. This is not a locked quote. {detail}",
        "hy": "Նախնական գնահատականը, մինչև տեղանքի ստուգումն ու արհեստանոցի վերանայումը, {range} է։ Սա վերջնական գին չէ։ {detail}",
        "fa": "برآورد تقریبی، پیش از بررسی محل و بازبینی کارگاه، {range} است. این قیمت قطعی نیست. {detail}",
        "es": "Un estimado aproximado, antes de revisar el sitio y el taller, es {range}. No es un precio cerrado. {detail}",
    },
    "ballpark_short": {
        "en": "A rough estimate before a site check and shop review is {range}. This is not a locked quote.",
        "hy": "Նախնական գնահատականը, մինչև տեղանքի ստուգումն ու արհեստանոցի վերանայումը, {range} է։ Սա վերջնական գին չէ։",
        "fa": "برآورد تقریبی، پیش از بررسی محل و بازبینی کارگاه، {range} است. این قیمت قطعی نیست.",
        "es": "Un estimado aproximado, antes de revisar el sitio y el taller, es {range}. No es un precio cerrado.",
    },
    "detail_gate": {
        "en": "{style}, clear opening {width} ft by {height} ft. Shop fabrication only.",
        "hy": "{style}՝ բացվածք {width} ֆուտ × {height} ֆուտ։ Միայն արհեստանոցային պատրաստում։",
        "fa": "{style}، دهانه {width} فوت در {height} فوت. فقط ساخت در کارگاه.",
        "es": "{style}, vano libre de {width} ft por {height} ft. Solo fabricación en taller.",
    },
    "detail_railing": {
        "en": "{style}, {length} ft long at {height} ft high. Shop fabrication only.",
        "hy": "{style}՝ {length} ֆուտ երկարություն, {height} ֆուտ բարձրություն։ Միայն արհեստանոցային պատրաստում։",
        "fa": "{style}، طول {length} فوت و ارتفاع {height} فوت. فقط ساخت در کارگاه.",
        "es": "{style}, {length} ft de largo y {height} ft de alto. Solo fabricación en taller.",
    },
    "detail_stair": {
        "en": "{style}, {steps} steps, {width} in wide. Shop fabrication only.",
        "hy": "{style}՝ {steps} աստիճան, {width} դյույմ լայնք։ Միայն արհեստանոցային պատրաստում։",
        "fa": "{style}، {steps} پله، عرض {width} اینچ. فقط ساخت در کارگاه.",
        "es": "{style}, {steps} escalones, {width} pulgadas de ancho. Solo fabricación en taller.",
    },
    "detail_patio": {
        "en": "{style}, {area} sq ft, {span} ft span. Shop fabrication only.",
        "hy": "{style}՝ {area} քառ. ֆուտ, {span} ֆուտ թռիչք։ Միայն արհեստանոցային պատրաստում։",
        "fa": "{style}، {area} فوت مربع، دهانه {span} فوت. فقط ساخت در کارگاه.",
        "es": "{style}, {area} pies cuadrados, claro de {span} ft. Solo fabricación en taller.",
    },
    "quantity": {
        "en": "Quantity {count}.",
        "hy": "Քանակ {count}։",
        "fa": "تعداد {count}.",
        "es": "Cantidad {count}.",
    },
}

CAPTIONS = {
    "modern_horizontal_gate": {
        "en": "Modern horizontal gate",
        "hy": "Ժամանակակից հորիզոնական դարպաս",
        "fa": "دروازه افقی مدرن",
        "es": "Portón horizontal moderno",
    },
    "ornamental_iron_gate": {
        "en": "Ornamental iron gate",
        "hy": "Դեկորատիվ երկաթե դարպաս",
        "fa": "دروازه فرفورژه",
        "es": "Portón de hierro ornamental",
    },
    "stair_railing": {
        "en": "Stair railing",
        "hy": "Աստիճանի բազրիք",
        "fa": "نرده پله",
        "es": "Baranda de escalera",
    },
    "outdoor_patio_frame": {
        "en": "Outdoor patio frame",
        "hy": "Արտաքին պատշգամբի շրջանակ",
        "fa": "قاب فضای باز پاسیو",
        "es": "Marco de patio exterior",
    },
}

STYLE_NAMES: dict[str, dict[str, str]] = {
    "vertical_bar_single": {
        "en": "Vertical-bar single swing",
        "hy": "Ուղղաձիգ ճաղերով միաթև դարպաս",
        "fa": "دروازه تک‌لنگه با میله عمودی",
        "es": "Portón de una hoja con barrotes verticales",
    },
    "vertical_bar_double": {
        "en": "Vertical-bar double swing",
        "hy": "Ուղղաձիգ ճաղերով երկթև դարպաս",
        "fa": "دروازه دو لنگه با میله عمودی",
        "es": "Portón de dos hojas con barrotes verticales",
    },
    "ornamental_single": {
        "en": "Ornamental single swing",
        "hy": "Դեկորատիվ միաթև դարպաս",
        "fa": "دروازه تک‌لنگه فرفورژه",
        "es": "Portón ornamental de una hoja",
    },
    "ornamental_double": {
        "en": "Ornamental double swing",
        "hy": "Դեկորատիվ երկթև դարպաս",
        "fa": "دروازه دو لنگه فرفورژه",
        "es": "Portón ornamental de dos hojas",
    },
    "vertical_picket": {
        "en": "Vertical picket railing",
        "hy": "Ուղղաձիգ ճաղերով բազրիք",
        "fa": "نرده با میله عمودی",
        "es": "Baranda de barrotes verticales",
    },
    "horizontal_picket": {
        "en": "Horizontal picket railing",
        "hy": "Հորիզոնական ճաղերով բազրիք",
        "fa": "نرده با میله افقی",
        "es": "Baranda de barrotes horizontales",
    },
    "cable": {
        "en": "Cable railing",
        "hy": "Ճոպանային բազրիք",
        "fa": "نرده کابلی",
        "es": "Baranda de cable",
    },
    "ornamental": {
        "en": "Ornamental railing",
        "hy": "Դեկորատիվ բազրիք",
        "fa": "نرده فرفورژه",
        "es": "Baranda ornamental",
    },
    "mono_stringer": {
        "en": "Mono-stringer stair",
        "hy": "Մեկ հեծանով աստիճան",
        "fa": "پله تک‌تیر",
        "es": "Escalera de un larguero",
    },
    "channel_stringer": {
        "en": "Channel stringer stair",
        "hy": "Շվելլերային աստիճան",
        "fa": "پله با تیر ناودانی",
        "es": "Escalera de largueros en canal",
    },
    "spiral": {
        "en": "Spiral stair",
        "hy": "Պարուրաձև աստիճան",
        "fa": "پله مارپیچ",
        "es": "Escalera de caracol",
    },
    "floating": {
        "en": "Floating stair",
        "hy": "Կախովի աստիճան",
        "fa": "پله شناور",
        "es": "Escalera flotante",
    },
    "pergola": {
        "en": "Steel pergola",
        "hy": "Պողպատե պերգոլա",
        "fa": "پرگولای فولادی",
        "es": "Pérgola de acero",
    },
    "patio_cover": {
        "en": "Patio cover",
        "hy": "Պատշգամբի ծածկ",
        "fa": "سایبان پاسیو",
        "es": "Cubierta de patio",
    },
    "framing": {
        "en": "Outdoor frame",
        "hy": "Արտաքին շրջանակ",
        "fa": "قاب فضای باز",
        "es": "Marco exterior",
    },
    "perimeter_railing": {
        "en": "Perimeter railing",
        "hy": "Պարագծային բազրիք",
        "fa": "نرده پیرامونی",
        "es": "Baranda perimetral",
    },
}


def normalize_client(value: Any) -> str:
    """Return a supported client language code, or English."""
    if isinstance(value, str):
        code = value.strip().lower()
        if code in CLIENT_LANGUAGES:
            return code
    return "en"


def normalize_shop(value: Any) -> str:
    """Return English or Farsi for the shop review."""
    if isinstance(value, str):
        code = value.strip().lower()
        if code in SHOP_LANGUAGES:
            return code
    return "en"


def is_rtl(lang: str) -> bool:
    """Return whether this language reads right to left."""
    return lang == "fa"


def text(lang: str, key: str, **values: object) -> str:
    """Return one phrase in the requested language, with English as the backup."""
    row = PHRASES[key]
    template = row.get(lang) or row["en"]
    if not values:
        return template
    return template.format(**values)


def has_any(text_value: str, phrases: tuple[str, ...]) -> bool:
    """Return whether any phrase appears as its own word or phrase."""
    for phrase in phrases:
        if re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text_value, re.IGNORECASE):
            return True
    return False


def reply_language(lang: str) -> str:
    """Tell the model which language the client reply should use.

    The wording is fixed for the four supported codes. It is not shown to the browser.
    """
    names = {"en": "English", "hy": "Armenian", "fa": "Farsi", "es": "Spanish"}
    code = normalize_client(lang)
    return (
        f"Write the reply string in {names[code]}. "
        "Keep every JSON key in English. Do not invent a dollar amount."
    )


def caption(lang: str, key: str) -> str:
    """Return the sample-photo caption for one example category."""
    row = CAPTIONS.get(key, {})
    return row.get(lang) or row.get("en") or key


def style_name(lang: str, key: str) -> str:
    """Return a catalog style name in the requested language."""
    row = STYLE_NAMES.get(key)
    if row:
        return row.get(lang) or row["en"]
    return key.replace("_", " ")


def _num(value: object) -> str:
    """Format a measurement without a trailing decimal when it is whole."""
    number = float(value)  # type: ignore[arg-type]
    if number.is_integer():
        return str(int(number))
    return f"{number:g}"


def describe(lang: str, facts: dict[str, Any]) -> str:
    """Describe the takeoff behind a rough ballpark, without a price."""
    code = normalize_client(lang) if lang in CLIENT_LANGUAGES else normalize_shop(lang)
    project = facts.get("project")
    style = style_name(code, str(facts.get("style_key") or ""))
    if not facts.get("style_key") and isinstance(project, str):
        style = text(code, f"project_{project}") if f"project_{project}" in PHRASES else style
    detail = ""
    if project == "gate" and facts.get("width_ft") and facts.get("height_ft"):
        detail = text(code, "detail_gate", style=style, width=_num(facts["width_ft"]), height=_num(facts["height_ft"]))
    elif project == "railing" and facts.get("length_ft") and facts.get("height_ft"):
        detail = text(
            code,
            "detail_railing",
            style=style,
            length=_num(facts["length_ft"]),
            height=_num(facts["height_ft"]),
        )
    elif project == "stair" and facts.get("steps") and facts.get("width_in"):
        detail = text(code, "detail_stair", style=style, steps=_num(facts["steps"]), width=_num(facts["width_in"]))
    elif project == "patio" and facts.get("area_sq_ft") and facts.get("span_ft"):
        detail = text(code, "detail_patio", style=style, area=_num(facts["area_sq_ft"]), span=_num(facts["span_ft"]))
    elif style and style != " ":
        detail = style
    quantity = facts.get("quantity")
    if detail and isinstance(quantity, (int, float)) and float(quantity) > 1:
        detail = f"{detail} {text(code, 'quantity', count=_num(quantity))}"
    return detail


def ballpark_sentence(lang: str, range_label: str, facts: dict[str, Any]) -> str:
    """Return the rough-estimate sentence. The range is never a single price."""
    code = normalize_client(lang)
    detail = describe(code, facts)
    if detail:
        return text(code, "ballpark", range=range_label, detail=detail)
    return text(code, "ballpark_short", range=range_label)


def unit_question(lang: str, width: float, height: float, *, modern: bool) -> str:
    """Ask whether a small-gate size was meant in inches."""
    code = normalize_client(lang)
    look = text(code, "modern_look") if modern else ""
    return text(code, "unit_question", look=look, width=_num(width), height=_num(height))


def note_examples(reply: str, examples: list[str], asked: bool, lang: str) -> str:
    """Add one short line when a sample photo belongs with this reply."""
    if not asked:
        return reply
    code = normalize_client(lang)
    lowered = reply.lower()
    if not examples:
        if any(marker in lowered for marker in WHICH_MARKERS):
            return reply
        return f"{reply} {text(code, 'which_sample')}".strip()
    if any(marker in lowered for marker in SAMPLE_MARKERS):
        return reply
    key = "samples_many" if len(examples) > 1 else "sample_one"
    return f"{reply} {text(code, key)}".strip()


def present_error(message: str, lang: str) -> str:
    """Show a known chat limit in the client's language."""
    code = normalize_client(lang)
    seconds = re.search(r"(\d+)\s+seconds", message)
    if seconds or "before sending more" in message.lower():
        count = seconds.group(1) if seconds else "15"
        return text(code, "rate_limit", seconds=count)
    if "questions" in message.lower() and "reached" in message.lower():
        return text(code, "turn_limit")
    return message


CONTACT_MARKERS = (
    "site check",
    "shop review",
    "site measurement",
    "checks the site",
    "check the site",
    "measure the site",
    "տեղանք",
    "չափում",
    "վերանայ",
    "տեխնիկ",
    "بررسی محل",
    "بازبینی کارگاه",
    "اندازه‌گیری",
    "تکنسین",
    "revisar el sitio",
    "revise el sitio",
    "medidas del sitio",
    "técnico",
    "tecnico",
)


def attach_shop_contact(reply: str, lang: str, phone: str, email: str) -> str:
    """Add the shop phone and email when a reply asks for a site check.

    Args:
        reply: The client-facing answer.
        lang: Client language code.
        phone: Shop phone from settings. Empty skips the sentence.
        email: Shop email from settings. Empty skips the sentence.

    Returns:
        The same reply, with one scheduling sentence when a visit is mentioned.
    """
    number = phone.strip()
    address = email.strip()
    if not number or not address or number in reply or address in reply:
        return reply
    lowered = reply.lower()
    if not any(marker in lowered for marker in CONTACT_MARKERS):
        return reply
    line = text(normalize_client(lang), "schedule_technician", phone=number, email=address)
    return f"{reply.rstrip()} {line}"


def shop_alert(
    lang: str,
    reference: str,
    name: str,
    contact: str,
    client_language: str,
    facts: dict[str, Any],
    questions: list[str],
    note: str,
) -> str:
    """Format one request as a text the shop can read in English or Farsi."""
    code = normalize_shop(lang)
    lines = [text(code, "alert_heading", reference=reference or "").strip()]
    if name.strip():
        lines.append(text(code, "alert_customer", name=name.strip()))
    if contact.strip():
        lines.append(text(code, "alert_contact", contact=contact.strip()))
    summary = shop_summary(code, facts, client_language)
    if summary:
        lines.append(summary)
    visible = [item.strip() for item in questions if isinstance(item, str) and item.strip()]
    if visible:
        lines.append(text(code, "customer_questions"))
        lines.extend(f"- {item}" for item in visible)
    if note.strip():
        lines.append(text(code, "extra_note"))
        lines.append(note.strip())
    return "\n".join(lines)


def shop_sms_brief(
    lang: str,
    reference: str,
    name: str,
    contact: str,
    facts: dict[str, Any],
    questions: list[str],
    note: str,
) -> str:
    """Return a short shop text for a carrier email gateway.

    Carrier email-to-SMS often drops long messages. The copy box still has
    the full summary.
    """
    code = normalize_shop(lang)
    parts = [text(code, "alert_heading", reference=reference or "").strip()]
    who = " ".join(item for item in (name.strip(), contact.strip()) if item)
    if who:
        parts.append(who)
    project = facts.get("project") if isinstance(facts, dict) else None
    if isinstance(project, str) and project in {"gate", "stair", "patio", "railing"}:
        parts.append(text(code, f"project_{project}"))
    range_label = facts.get("range") if isinstance(facts, dict) else None
    if isinstance(range_label, str) and range_label.startswith("$"):
        parts.append(range_label)
    first = next((item.strip() for item in questions if isinstance(item, str) and item.strip()), "")
    if first:
        parts.append(first[:140])
    if note.strip():
        parts.append(note.strip()[:80])
    return "\n".join(parts)[:320].rstrip()


def sms_link(phone: str, body: str) -> str:
    """Build an sms: link for the shop phone and a formatted summary.

    A ten-digit US number gains the country code. The body stays short enough
    for a text-message link. An empty phone returns an empty string.
    """
    digits = re.sub(r"\D", "", phone)
    if len(digits) == 10:
        digits = "1" + digits
    if not digits:
        return ""
    clipped = body if len(body) <= 900 else body[:900].rstrip() + "…"
    return f"sms:+{digits}?body={quote(clipped)}"


def shop_summary(lang: str, facts: dict[str, Any], client_language: str) -> str:
    """Build the shop card in English or Farsi from stored measurements."""
    code = normalize_shop(lang)
    if not facts:
        return ""
    lines = [text(code, "shop_summary_title")]
    if client_language in LANGUAGE_LABELS:
        lines.append(text(code, "customer_language", language=LANGUAGE_LABELS[client_language]))
    project = facts.get("project")
    if isinstance(project, str) and project in {"gate", "stair", "patio", "railing"}:
        lines.append(text(code, "project_line", project=text(code, f"project_{project}")))
    detail = describe(code, facts)
    if detail:
        lines.append(detail)
    range_label = facts.get("range")
    if isinstance(range_label, str) and range_label.startswith("$"):
        lines.append(text(code, "ballpark_short", range=range_label))
    return "\n".join(lines)
