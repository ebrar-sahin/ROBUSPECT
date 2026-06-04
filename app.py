import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent

st.set_page_config(page_title="Robuspect MLSecOps Platform", layout="wide")

# Başlık ve Açıklamalar
st.title("🛡️ Robuspect: Canlı Güvenilirlik Analiz Simülatörü")
st.subheader("CNN Tabanlı Yapay Zekâ Sistemlerinde Dinamik Robustness Haritalandırılması")
st.markdown("---")

# Sol Panel: Kullanıcı Seçim Alanı (Girdiler)
with st.sidebar:
    st.header("⚙️ Simülasyon Girdileri")
    
    # 1. Saldırı Türü Seçimi
    attack_type = st.selectbox(
        "Saldırı Yöntemi Seçin:",
        ["FGSM (Fast Gradient Sign Method)", "PGD (Projected Gradient Descent)"]
    )
    
    # 2. Pertürbasyon Türü (Norm) Seçimi
    norm_type = st.selectbox(
        "Pertürbasyon Normu (Evasion Norm):",
        ["L-infinity (L_inf)", "L-2 Norm"]
    )
    
    # 3. Epsilon Değeri Ayarı (Slider)
    eps_value = st.slider(
        "Gürültü Şiddeti (Epsilon ε):",
        min_value=0.01,
        max_value=0.50,
        value=0.15,
        step=0.01
    )
    
    st.markdown("---")
    st.markdown("**Akademik Künye:**\nAraştırmacı: Ebrar Şahin\nDanışman: Dr. Öğr. Üyesi Salih Sarp")

# Ana Panel Tasarımı
col1, col2 = st.columns([1, 1])

# Colab oturumundaki x_test_np, y_test_np ve classifier nesnelerini
# Streamlit ortamında simüle edebilmek için üst katmandan çağırıyoruz.
# (Yazılımın entegrasyon bütünlüğü için doğrudan üst hafızayı okur)
from __main__ import classifier, x_test_np, y_test_np, guaranteed_baseline_acc

with col1:
    st.markdown("### 📊 Canlı Risk Değerlendirmesi")
    st.metric(label="Modelin Temiz Doğruluğu (Baseline)", value=f"%{guaranteed_baseline_acc:.2f}")
    
    # Tetikleme Butonu
    run_btn = st.button("🚀 SİMÜLASYONU BAŞLAT", use_container_width=True)
    
    st.markdown("""
    **Kullanım Talimatı:** Sol panelden test etmek istediğiniz siber tehdit parametrelerini belirleyin ve yukarıdaki butona basın. 
    Algoritma, test seti imajlarını beyaz kutu (white-box) senaryosunda manipüle ederek yapay zekanın zafiyet sınırlarını canlı hesaplayacaktır.
    """)

# Butona basıldığında çalışacak dinamik motor mekanizması
if run_btn:
    with st.spinner("Siber saldırı simüle ediliyor ve hata matrisi üretiliyor..."):
        
        # Norm Ayarı
        art_norm = np.inf if norm_type == "L-infinity (L_inf)" else 2
        
        # Saldırı Motorunu Yapılandırma
        if attack_type.startswith("FGSM"):
            attack_engine = FastGradientMethod(estimator=classifier, eps=eps_value)
        else:
            step_size = eps_value / 10 if art_norm == np.inf else (eps_value / 5)
            attack_engine = ProjectedGradientDescent(
                estimator=classifier, norm=art_norm, eps=eps_value, eps_step=step_size, max_iter=10
            )
        
        # Canlı Saldırı Üretimi ve Tahmin
        x_adv = attack_engine.generate(x=x_test_np)
        adv_preds = np.argmax(classifier.predict(x_adv), axis=1)
        robust_acc = np.sum(adv_preds == y_test_np) / len(y_test_np) * 100
        
        # Sonuç Metrik Kartı
        with col1:
            st.markdown("---")
            st.metric(label="Saldırı Altındaki Doğruluk (Robust Accuracy)", value=f"%{robust_acc:.2f}", delta=f"- %{guaranteed_baseline_acc - robust_acc:.2f} Düşüş")
            
            # Dinamik Kararsızlık Bölgesi Analizi Raporlaması
            if (guaranteed_baseline_acc - robust_acc) >= 10.0:
                st.error(f"🚨 KRİTİK SEVİYE: Seçilen {eps_value} epsilon değeri modelin 'Kararsızlık Bölgesi' sınırları içerisindedir!")
            else:
                st.success("🟢 GÜVENLİ SEVİYE: Model bu gürültü şiddetine karşı kabul edilebilir bir direnç gösteriyor.")
        
        # Confusion Matrix Çizdirme ve Sağ Sütuna Basma
        with col2:
            st.markdown(f"### 🧮 Akademik Hata Matrisi ({attack_type})")
            cm = confusion_matrix(y_test_np, adv_preds, labels=list(range(10)))
            
            fig, ax = plt.subplots(figsize=(6, 4.5))
            color_map = "Blues" if attack_type.startswith("FGSM") else "Reds"
            sns.heatmap(cm, annot=True, fmt='d', cmap=color_map, xticklabels=range(10), yticklabels=range(10), cbar=False, ax=ax)
            ax.set_xlabel('Yapay Zekâ Tahmini')
            ax.set_ylabel('Gerçek Rakam')
            plt.tight_layout()
            
            st.pyplot(fig)
else:
    with col2:
        st.info("💡 Lütfen simülasyonu başlatmak ve akademik hata matrisini (Confusion Matrix) canlı görüntülemek için soldaki butona basın.")
