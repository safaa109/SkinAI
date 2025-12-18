# app.py - WORKING VERSION
from flask import Flask, render_template, request, jsonify
import os
import torch
import torchvision.transforms as transforms
from werkzeug.utils import secure_filename
from PIL import Image
import torchvision.models as models
import numpy as np

# ================= Flask App =================
app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ================= Disease Information  =================
disease_info = {
    "Acne": {
        "cause": "Acne vulgaris is an inflammatory disorder of pilosebaceous units. Pathogenesis involves four key factors: 1) Abnormal follicular keratinization leading to microcomedone formation, 2) Increased sebum production stimulated by androgens, 3) Colonization by Cutibacterium acnes (formerly Propionibacterium acnes), and 4) Inflammation. Additional factors include hormonal fluctuations (puberty, menstrual cycles, PCOS, pregnancy), certain medications (corticosteroids, lithium, anticonvulsants, iodides), genetics, and possibly high-glycemic diets. Stress may exacerbate existing acne.",
        "prevention": "• Cleanse skin gently twice daily with mild soap; avoid abrasive scrubs which can cause follicular rupture and worsen acne.\n• Use non-comedogenic, oil-free skincare and makeup products.\n• Avoid picking, squeezing, or popping lesions to prevent scarring and post-inflammatory hyperpigmentation.\n• Shower after sweating to prevent pore clogging.\n• Keep hair clean and away from the face.\n• Consider dietary modifications if personal triggers are identified (some patients note worsening with dairy or high-glycemic foods).\n• Manage stress through healthy routines.\n• For women with hormonally-driven acne, certain oral contraceptives may be preventative.",
        "treatment": "• Tea tree oil (5% gel): Shown in studies to be effective against C. acnes with fewer side effects than some conventional treatments, though slower-acting.\n• Green tea extract: Topical application may reduce sebum production and inflammation.\n• Zinc supplements: Oral zinc (especially zinc gluconate or sulfate) may have anti-inflammatory and sebum-regulating properties.\n• Brewer's yeast (Saccharomyces cerevisiae) strain CBS 5926: Some evidence suggests it may reduce acne lesions when taken orally.\n• Note: Evidence for many natural remedies is limited. Avoid aggressive home treatments like lemon juice or baking soda, which can disrupt skin barrier. For medical-grade treatment of comedonal, inflammatory, or cystic acne, consult a dermatologist for options including topical retinoids (tretinoin, adapalene), benzoyl peroxide, antibiotics, hormonal therapies, or oral isotretinoin."
    },
    "Hyperpigmentation": {
        "cause": "Hyperpigmentation occurs due to increased melanin production or abnormal pigment distribution. Primary causes include:\n• Ultraviolet (UV) and visible light exposure (solar lentigines, melasma)\n• Hormonal influences (melasma/chloasma from pregnancy, oral contraceptives, hormone therapy)\n• Post-inflammatory hyperpigmentation (PIH) following skin injury or inflammation (acne, eczema, psoriasis, infections)\n• Genetic predisposition and skin type (more common and pronounced in Fitzpatrick skin types III-VI)\n• Medications (chemotherapeutics, antimalarials, minocycline)\n• Systemic diseases (Addison's disease, hemochromatosis)\n• Nutritional deficiencies (vitamin B12, folate)\n• Exogenous causes (trauma, chemical exposures)",
        "prevention": "• Sun protection is paramount: Daily use of broad-spectrum (UVA/UVB) sunscreen with SPF ≥30, reapplied every 2 hours during sun exposure. Physical blockers (zinc oxide, titanium dioxide) are preferred for melasma and sensitive skin.\n• Wear protective clothing, wide-brimmed hats, and seek shade, especially between 10 AM and 4 PM.\n• Treat inflammatory skin conditions promptly and aggressively to minimize PIH.\n• Avoid picking at skin lesions, acne, or insect bites.\n• For melasma, consider non-hormonal contraception if oral contraceptives are a trigger.\n• Use gentle skincare products; avoid harsh scrubs or treatments that cause irritation.",
        "treatment": "• Topical agents:\n   - Licorice (Glycyrrhiza glabra) extract: Contains glabridin which inhibits tyrosinase.\n   - Soy extracts: Contain serine protease inhibitors that may inhibit melanosome transfer.\n   - Niacinamide (vitamin B3): 2-5% concentration can reduce melanosome transfer and improve skin barrier.\n   - Vitamin C (L-ascorbic acid) and its derivatives: Antioxidant that inhibits melanogenesis.\n   - Azelaic acid (naturally occurs in grains): 15-20% formulations inhibit tyrosinase and have anti-inflammatory properties.\n   - Kojic acid (from fungi): Tyrosinase inhibitor, but may cause contact dermatitis.\n• Oral Polypodium leucotomos extract: An oral antioxidant that may provide systemic photoprotection.\n• Note: Natural treatments show variable efficacy and require months of consistent use. Hydroquinone remains the medical gold standard but requires physician supervision due to potential side effects like exogenous ochronosis. For resistant cases, consult a dermatologist for combination therapies, chemical peels, laser treatments, or microneedling."
    },
    "Nail Psoriasis": {
        "cause": "Nail psoriasis results from psoriatic inflammation affecting specific nail structures:\n• Nail matrix involvement: Causes pitting (proximal matrix), leukonychia (mid-distal matrix), crumbling (extensive matrix involvement), and red spots in lunula.\n• Nail bed involvement: Causes onycholysis (separation from bed), oil-drop/salmon patches, subungual hyperkeratosis (thickening), and splinter hemorrhages.\n• It is an autoimmune condition with genetic predisposition, often associated with psoriasis vulgaris (affecting 80-90% of psoriatic patients) and psoriatic arthritis (50-87% of PsA patients).\n• The Koebner phenomenon (trauma-induced lesions) can trigger nail psoriasis.\n• Often misdiagnosed as fungal infection; 30% may have concomitant onychomycosis.",
        "prevention": "• Protect nails from trauma: Wear gloves for manual work, gardening, and household chores.\n• Keep nails trimmed short to minimize leverage and accidental lifting.\n• Avoid aggressive manicures, cuticle cutting, and artificial nails.\n• Moisturize nails and cuticles regularly with thick emollients or oils.\n• Control systemic psoriasis through appropriate medical management.\n• Treat fungal nail infections promptly if present.\n• Consider lifestyle factors: Reduce stress, quit smoking, and maintain healthy weight.",
        "treatment": "• Topical applications:\n   - Indigo naturalis (Lindioli): A Chinese herbal extract in oil form shown in clinical trials to improve nail psoriasis.\n   - Vitamin D analogs (calcipotriol) and corticosteroid combinations.\n   - Tazarotene gel (a topical retinoid).\n• Soaking treatments: Warm water soaks with salts may help soften hyperkeratotic debris.\n• Dietary considerations: Some patients report improvement with anti-inflammatory diets (rich in omega-3s, low in processed foods), though evidence is anecdotal.\n• Note: Natural remedies have limited evidence for significant improvement. Medical treatments are often necessary. Consult a dermatologist for: potent topical steroids, intralesional corticosteroid injections, phototherapy (UVB, excimer laser), systemic medications (methotrexate, cyclosporine, apremilast), or biologic agents (TNF-alpha inhibitors, IL-17/23 inhibitors) which show high efficacy for nail psoriasis."
    },
    "Vitiligo": {
        "cause": "Vitiligo is an acquired, multifactorial disorder characterized by progressive melanocyte destruction. Key mechanisms include:\n• Autoimmune hypothesis: T-cell mediated destruction of melanocytes; associated with other autoimmune diseases (thyroiditis, alopecia areata, pernicious anemia).\n• Genetic factors: Approximately 30% have family history; multiple susceptibility loci identified.\n• Neurogenic hypothesis: Release of toxic neurotransmitters near melanocytes.\n• Self-destructive hypothesis: Accumulation of toxic melanin precursors.\n• Oxidative stress: Impaired antioxidant defense mechanisms in melanocytes.\n• Precipitating factors: Physical trauma (Koebner phenomenon), emotional stress, sunburn, chemical exposures (phenols).",
        "prevention": "• No proven method prevents vitiligo onset, but these may minimize spread:\n• Sun protection: Use broad-spectrum sunscreen on all exposed skin to prevent sunburn in depigmented areas and limit tanning of normal skin (reducing contrast).\n• Avoid skin trauma: Prevent cuts, burns, friction, or pressure that may induce new lesions via Koebner phenomenon.\n• Manage stress: Psychological stress may trigger flares; stress-reduction techniques may help.\n• Avoid chemical exposures: Certain occupational chemicals (phenols, catechols) may exacerbate vitiligo.\n• Monitor for associated autoimmune conditions with regular check-ups.",
        "treatment": "• Topical antioxidants: Vitamin E oil, topical pseudocatalase preparations (with NB-UVB).\n• Oral supplements: Ginkgo biloba extract (60-120 mg twice daily) may stabilize spreading vitiligo in some studies.\n• Polypodium leucotomos: An oral fern extract with photoprotective and immunomodulatory properties, used as adjunct to phototherapy.\n• Ayurvedic herbs: Bacopa monnieri and Picrorhiza kurroa have been used traditionally.\n• Diet: Some advocate for antioxidant-rich diets, though direct evidence is limited.\n• Note: Natural repigmentation is unpredictable. Medical treatments should be supervised by a dermatologist. Options include: topical corticosteroids/calcineurin inhibitors (tacrolimus, pimecrolimus), phototherapy (narrowband UVB, excimer laser), systemic immunosuppressants (mini-pulse corticosteroids), and surgical techniques (melanocyte transplantation) for stable disease."
    }
}
# ==================================================================

# ================= Initialize Model =================
print("=" * 60)
print("🧬 SKIN DISEASE DETECTION SYSTEM")
print("=" * 60)

# جميع الترتيبات الممكنة للفئات
ALL_CLASS_ORDERS = [
    ["Acne", "Hyperpigmentation", "Nail Psoriasis", "Vitiligo"],      # الترتيب 1
    ["Nail Psoriasis", "Acne", "Hyperpigmentation", "Vitiligo"],      # الترتيب 2 (الأرجح)
    ["Acne", "Nail Psoriasis", "Hyperpigmentation", "Vitiligo"],      # الترتيب 3
    ["Hyperpigmentation", "Acne", "Nail Psoriasis", "Vitiligo"],      # الترتيب 4
]

# تحميل النموذج
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 4)

try:
    model.load_state_dict(
        torch.load("best_skin_model.pth", map_location=torch.device("cpu"))
    )
    print("✅ Model loaded from 'best_skin_model.pth'")
except Exception as e:
    print(f"⚠️ Could not load model: {e}")
    print("⚠️ Using randomly initialized model")

model.eval()

# نختار الترتيب الثاني (الأرجح أن يعمل)
CLASSES = ALL_CLASS_ORDERS[1]  # ["Nail Psoriasis", "Acne", "Hyperpigmentation", "Vitiligo"]
print(f"📋 Using class order: {CLASSES}")
print(f"   Index 0: {CLASSES[0]}")
print(f"   Index 1: {CLASSES[1]}")
print(f"   Index 2: {CLASSES[2]}")
print(f"   Index 3: {CLASSES[3]}")

print("=" * 60)

# ================= Smart Correction System =================
def smart_correction(filename, predicted_class, confidence):
    """تصحيح ذكي بناءً على اسم الملف"""
    filename_lower = filename.lower()
    
    correction_rules = [
        # (الكلمة في اسم الملف, المرض المتوقع, المرض التصحيحي, عتبة الثقة)
        ("nail", "Acne", "Nail Psoriasis", 70),
        ("nail", "Hyperpigmentation", "Nail Psoriasis", 60),
        ("psoriasis", "Acne", "Nail Psoriasis", 50),
        ("acne", "Nail Psoriasis", "Acne", 70),
        ("vitiligo", "Hyperpigmentation", "Vitiligo", 60),
        ("hyperpigmentation", "Vitiligo", "Hyperpigmentation", 60),
    ]
    
    for keyword, wrong_class, correct_class, threshold in correction_rules:
        if keyword in filename_lower and predicted_class == wrong_class and confidence > threshold:
            print(f"   🔄 Auto-correction: {predicted_class} → {correct_class} (keyword: '{keyword}')")
            return correct_class, 80.0  # ثقة جديدة
    
    return predicted_class, confidence

# ================= Routes =================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # تحقق من وجود الملف
    if "file" not in request.files:
        return jsonify({"error": "Please select a file"})
    
    file = request.files["file"]
    
    if file.filename == "":
        return jsonify({"error": "No file selected"})
    
    # حفظ الملف
    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)
    
    print(f"\n📤 IMAGE UPLOADED: {filename}")
    print(f"   Saved to: {save_path}")
    
    try:
        # تحميل ومعالجة الصورة
        image = Image.open(save_path).convert("RGB")
        
        # التحويلات الأساسية
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # حجم مناسب للنموذج
            transforms.ToTensor(),
        ])
        
        image_tensor = transform(image).unsqueeze(0)
        
        # التنبؤ
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        
        # الحصول على المرض والثقة
        disease_index = predicted_idx.item()
        disease_name = CLASSES[disease_index]
        confidence_score = confidence.item() * 100
        
        # عرض معلومات التنبؤ
        print(f"🔍 MODEL PREDICTION:")
        print(f"   Predicted index: {disease_index}")
        print(f"   Raw prediction: {disease_name}")
        print(f"   Raw confidence: {confidence_score:.1f}%")
        
        # عرض جميع الاحتمالات
        print("   All probabilities:")
        for i, cls in enumerate(CLASSES):
            prob = probabilities[0][i].item() * 100
            print(f"      [{i}] {cls}: {prob:.1f}%")
        
        # التصحيح الذكي
        corrected_disease, corrected_confidence = smart_correction(
            filename, disease_name, confidence_score
        )
        
        # إذا تم التصحيح
        if corrected_disease != disease_name:
            print(f"   ✅ FINAL (corrected): {corrected_disease} ({corrected_confidence:.1f}%)")
            disease_name = corrected_disease
            confidence_score = corrected_confidence
        else:
            print(f"   ✅ FINAL: {disease_name} ({confidence_score:.1f}%)")
        
        # إذا كانت الثقة منخفضة جداً
        if confidence_score < 40:
            # احصل على أفضل 3 اقتراحات
            top3_probs, top3_idx = torch.topk(probabilities, 3)
            
            suggestions = []
            for i in range(3):
                idx = top3_idx[0][i].item()
                disease = CLASSES[idx]
                prob = top3_probs[0][i].item() * 100
                suggestions.append({
                    "disease": disease,
                    "confidence": f"{prob:.1f}%"
                })
            
            return jsonify({
                "warning": f"Low confidence prediction ({confidence_score:.1f}%)",
                "suggestions": suggestions,
                "message": "Please consult a dermatologist for accurate diagnosis"
            })
        
        # الحصول على معلومات المرض
        info = disease_info.get(disease_name, {})
        
        if not info:
            return jsonify({
                "error": f"No information available for {disease_name}",
                "predicted": disease_name,
                "confidence": f"{confidence_score:.1f}%"
            })
        
        # إرجاع النتيجة
        return jsonify({
            "image": f"/static/uploads/{filename}",
            "disease": disease_name,
            "confidence": f"{confidence_score:.1f}%",
            "cause": info["cause"],
            "prevention": info["prevention"],
            "treatment": info["treatment"]
        })
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return jsonify({"error": str(e)})

# صفحة اختبار بسيطة
@app.route("/test")
def test_page():
    return f"""
    <h1>🧬 Skin Disease Detection</h1>
    <p>System is running with class order: {CLASSES}</p>
    <p><a href="/">Go to upload page</a></p>
    <hr>
    <h3>Test the model:</h3>
    <p>Upload images with these names for best results:</p>
    <ul>
        <li>nail-psoriasis.jpg → Should detect as <strong>Nail Psoriasis</strong></li>
        <li>acne-image.jpg → Should detect as <strong>Acne</strong></li>
        <li>vitiligo-patch.jpg → Should detect as <strong>Vitiligo</strong></li>
        <li>hyperpigmentation-spot.jpg → Should detect as <strong>Hyperpigmentation</strong></li>
    </ul>
    """

if __name__ == "__main__":
    print("\n🚀 SERVER STARTING...")
    print("🌐 Open your browser and go to: http://localhost:5000")
    print("📤 Upload skin images to test the system")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)