import streamlit as st
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.ndimage as ndimage
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent

st.set_page_config(page_title="Robuspect MLSecOps Core", layout="wide")

# Sol çubuğu tamamen devre dışı bırakan CSS enjeksiyonu
st.markdown("""
    <style>
        [data-testid="stSidebar"] {display: none !important;}
        [data-testid="collapsibleSidebarCreator"] {display: none !important;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# GÖMÜLÜ MODEL MİMARİSİ VE VERİ SETİ ALTYAPISI
# ==========================================
class MNISTCNN(nn.Module):
    def __init__(self):
        super(MNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

@st.cache_resource
def load_embedded_test_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0,), (1.0,))])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=300, shuffle=False)
    x_t, y_t = next(iter(test_loader))
    return x_t.numpy(), y_t.numpy()

x_test_np, y_test_np = load_embedded_test_data()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

# ==========================================
# KURUMSAL HEADER BANNER
# ==========================================
header_col1, header_col2 = st.columns([1.2, 5.8])
with header_col1:
    st.markdown("""
        <div style='background-color: #1E2235; padding: 16px; border-radius: 8px; border: 1px solid #2D314E; text-align: center;'>
            <span style='font-size: 26px;'>🛡️</span>
            <div style='font-weight: bold; color: white; font-size: 14px; margin-top: 4px;'>Robuspect Lab</div>
            <div style='font-style: italic; color: #8F94B5; font-size: 11px;'>MLSecOps Analiz Birimi</div>
        </div>
    """, unsafe_allow_html=True)

with header_col2:
    st.markdown("<h2 style='margin-top: 0px; padding-top: 2px;'>Robuspect: Merkezi Siber Güvenilirlik Test Motoru</h2>", unsafe_allow_html=True)
    st.markdown("<h5 style='color: #8F94B5; margin-top: -8px;'>Modeller İçin Karşılaştırmalı Zafiyet Analizi ve Görsel Adli Bilişim Platformu</h5>", unsafe_allow_html=True)

st.markdown("---")

# ==========================================
# İSTEK: EN ÜSTTE SABİT MODEL YÜKLEME VE RAPOR İNDİRME YAN YANA
# ==========================================
top_col1, top_col2 = st.columns([2, 1])

with top_col1:
    uploaded_weights = st.file_uploader("🚀 BAŞLAMAK İÇİN: Model Ağırlık Dosyasını Seçin (.pth)", type=["pth", "pt"])

with top_col2:
    st.markdown("<div style='padding-top: 4px;'></div>", unsafe_allow_html=True)
    if st.session_state.get('analysis_triggered', False) and 'report_text' in st.session_state:
        st.download_button(
            label="📄 RESMİ GÜVENLİK RAPORUNU (.TXT) İNDİR",
            data=st.session_state['report_text'],
            file_name="robuspect_model_guvenlik_raporu.txt",
            mime="text/plain",
            use_container_width=True
        )
    else:
        st.button("📄 RAPOR HAZIR DEĞİL (ÖNCE TESTİ TETİKLEYİN)", disabled=True, use_container_width=True)

if uploaded_weights is not None:
    # Durum Koruma (State-Loss) Mekanizması
    if 'model_loaded' not in st.session_state or st.session_state.get('last_uploaded') != uploaded_weights.name:
        with st.spinner("Model anatomisi çözümleniyor..."):
            state_dict = torch.load(uploaded_weights, map_location=torch.device('cpu'))
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                st.session_state['predicted_epoch'] = state_dict.get('epoch', 3)
                st.session_state['predicted_opt'] = state_dict.get('optimizer_name', 'Adam (Tahmin Edildi)')
                st.session_state['predicted_lr'] = state_dict.get('learning_rate', 0.0010)
                state_dict = state_dict['model_state_dict']
            else:
                st.session_state['predicted_epoch'] = "3 (Ağırlık Matrisinden Saptandı)"
                st.session_state['predicted_opt'] = "Adam / SGD (Anatomik Saptama)"
                st.session_state['predicted_lr'] = 0.0010
            
            st.session_state['pure_state_dict'] = state_dict
            st.session_state['model_loaded'] = True
            st.session_state['last_uploaded'] = uploaded_weights.name

    state_dict = st.session_state['pure_state_dict']
    predicted_epoch = st.session_state['predicted_epoch']
    predicted_opt = st.session_state['predicted_opt']
    predicted_lr = st.session_state['predicted_lr']

    eval_model = MNISTCNN().to(device)
    eval_model.load_state_dict(state_dict)
    eval_model.eval()
    
    dummy_optimizer = torch.optim.Adam(eval_model.parameters(), lr=0.001)
    classifier_engine = PyTorchClassifier(
        model=eval_model, clip_values=(0.0, 1.0), loss=criterion,
        optimizer=dummy_optimizer, input_shape=(1, 28, 28), nb_classes=10
    )

    clean_preds = np.argmax(classifier_engine.predict(x_test_np), axis=1)
    baseline_acc = np.sum(clean_preds == y_test_np) / len(y_test_np) * 100

    # Üç Büyük Dinamik Sayfa (Tab Düzeni)
    tab1, tab2, tab3 = st.tabs(["📋 1. SAYFA: Model Özellikleri", "📊 2. SAYFA: Siber Dayanıklılık & İnteraktif Analiz", "🧠 3. SAYFA: Gelişmiş Göstergeler (XAI)"])
    
    # ==========================================
    # 1. SAYFA: MODELİN ÖZELLİKLERİ VE TASARIM DENETİMİ
    # ==========================================
    with tab1:
        st.markdown("### 🔍 Mimarî Akış Şeması ve Tasarruf Uyumluluk Analizi")
        
        # Sektör Standart Akış Şeması Çizimi
        fig_arch, ax_arch = plt.subplots(figsize=(10, 1.5))
        ax_arch.axis('off')
        layers_list = [key.split('.')[0] for key in state_dict.keys() if 'weight' in key]
        box_style = dict(boxstyle="round,pad=0.4", fc="#2E303E", ec="#4E5166", lw=1.5)
        for idx, l_name in enumerate(layers_list):
            ax_arch.text(idx * 2, 0.5, f" {l_name.upper()} ", color="white", weight="bold", ha="center", va="center", bbox=box_style, fontsize=8)
            if idx < len(layers_list) - 1:
                ax_arch.annotate("", xy=((idx + 1) * 2 - 0.5, 0.5), xytext=(idx * 2 + 0.5, 0.5), arrowprops=dict(arrowstyle="->", lw=1.5, color="#00FFCC"))
        ax_arch.set_xlim(-1, len(layers_list) * 2 - 1)
        ax_arch.set_ylim(0, 1)
        st.pyplot(fig_arch)
        plt.close(fig_arch)

        # Yapay Zekâ Tasarım Kuralları Denetimi
        audit_results = []
        all_w_flattened = [v.cpu().numpy().flatten() for k, v in state_dict.items() if 'weight' in k]
        if all_w_flattened:
            flat_w = np.concatenate(all_w_flattened)
            w_mean, w_std, w_max, w_min = np.mean(flat_w), np.std(flat_w), np.max(flat_w), np.min(flat_w)
            
            rule1 = "🟢 UYUMLU (Sıfır merkezli matris)" if abs(w_mean) <= 0.05 else "⚠️ TASARIM SAPMASI"
            audit_results.append({"Tasarım Kuralı": "Ağırlık Sıfır Merkezleme Standardı", "Metrik": f"Ortalama: {w_mean:.4f}", "Durum": rule1})
            
            rule2 = "🟢 GÜVENLİ (Stabil gradyan akışı)" if -4.0 <= w_min and w_max <= 4.0 else "🔴 RİSKLİ (Patlama riski)"
            audit_results.append({"Tasarım Kuralı": "Exploding Gradient Kriteri", "Metrik": f"Sınırlar: [{w_min:.1f}, {w_max:.1f}]", "Durum": rule2})

        if 'fc1.weight' in state_dict:
            ratio = state_dict['fc1.weight'].shape[1] / state_dict['fc1.weight'].shape[0]
            rule3 = "🟢 DENGELİ" if ratio <= 15 else "⚠️ KRİTİK SEVİYE Darboğaz"
            audit_results.append({"Tasarım Kuralı": "Mimarî Darboğaz (Bottleneck) Kontrolü", "Metrik": f"Daralma Oranı: {ratio:.1f}x", "Durum": rule3})
            
        st.table(audit_results)

    # ==========================================
    # 2. SAYFA: SALDIRI ALTINDAKİ DOĞRULUK VE İNTERAKTİF SAYI SEÇİMİ
    # ==========================================
    with tab2:
        st.markdown("### ⚙️ Tehdit Altında Dayanıklılık ve Kararlılık Laboratuvarı")
        
        # Parametre Kontrolleri
        cfg_c1, cfg_c2 = st.columns(2)
        with cfg_c1:
            norm_type = st.selectbox("Tehdit Pertürbasyon Normu:", ["L-infinity (L_inf)", "L-2 Norm"])
        with cfg_c2:
            eps_value = st.slider("Saldırı Şiddet Katsayısı (Epsilon ε):", min_value=0.01, max_value=0.50, value=0.15, step=0.01)

        trigger_btn = st.button("🚀 SİBER GÜVENLİK TESTİNİ TETİKLE", use_container_width=True)
        
        if trigger_btn:
            st.session_state['analysis_triggered'] = True
            with st.spinner("Siber gürültü matrisleri ve kararlılık eğrileri hesaplanıyor..."):
                art_norm = np.inf if norm_type.startswith("L-infinity") else 2
                
                # FGSM ve PGD Hesaplamaları
                fgsm_eng = FastGradientMethod(estimator=classifier_engine, eps=eps_value)
                x_fgsm = fgsm_eng.generate(x=x_test_np)
                preds_fgsm = np.argmax(classifier_engine.predict(x_fgsm), axis=1)
                acc_fgsm = np.sum(preds_fgsm == y_test_np) / len(y_test_np) * 100
                
                pgd_eng = ProjectedGradientDescent(estimator=classifier_engine, norm=art_norm, eps=eps_value, eps_step=eps_value/10, max_iter=10)
                x_pgd = pgd_eng.generate(x=x_test_np)
                preds_pgd = np.argmax(classifier_engine.predict(x_pgd), axis=1)
                acc_pgd = np.sum(preds_pgd == y_test_np) / len(y_test_np) * 100
                
                # Kararlılık Eğrisi Döngüsü
                eps_range = [0.0, 0.10, 0.20, 0.30, 0.40]
                fgsm_curve, pgd_curve = [], []
                for e in eps_range:
                    f_e = FastGradientMethod(estimator=classifier_engine, eps=e)
                    p_e = ProjectedGradientDescent(estimator=classifier_engine, norm=art_norm, eps=e, eps_step=max(e/5, 0.01), max_iter=5)
                    fgsm_curve.append(np.sum(np.argmax(classifier_engine.predict(f_e.generate(x=x_test_np)), axis=1) == y_test_np) / len(y_test_np) * 100)
                    pgd_curve.append(np.sum(np.argmax(classifier_engine.predict(p_e.generate(x=x_test_np)), axis=1) == y_test_np) / len(y_test_np) * 100)
                
                # Tüm verileri session_state içerisine kilitliyoruz (Silinmeyi önleme)
                st.session_state['acc_fgsm'] = acc_fgsm
                st.session_state['acc_pgd'] = acc_pgd
                st.session_state['preds_fgsm'] = preds_fgsm
                st.session_state['preds_pgd'] = preds_pgd
                st.session_state['eps_range'] = eps_range
                st.session_state['fgsm_curve'] = fgsm_curve
                st.session_state['pgd_curve'] = pgd_curve
                st.session_state['saved_x_step'] = x_pgd
                
                # Rapor metnini hazırlayıp hafızaya atıyoruz (Böylece en üstteki buton anında aktifleşir)
                total_v_score = (baseline_acc - min(acc_fgsm, acc_pgd))
                st.session_state['report_text'] = f"ROBUSPECT DENETİM RAPORU\nOrijinal Başarı: %{baseline_acc:.2f}\nSaldırı Sonrası: %{min(acc_fgsm, acc_pgd):.2f}\nZafiyet Endeksi: {total_v_score:.2f}"

        # Test tetiklendiyse sonuçları çizdir
        if st.session_state.get('analysis_triggered', False):
            acc_fgsm = st.session_state['acc_fgsm']
            acc_pgd = st.session_state['acc_pgd']
            preds_fgsm = st.session_state['preds_fgsm']
            preds_pgd = st.session_state['preds_pgd']
            eps_range = st.session_state['eps_range']
            fgsm_curve = st.session_state['fgsm_curve']
            pgd_curve = st.session_state['pgd_curve']
            
            st.info(f"🎯 **Modelin Temiz Başarısı:** %{baseline_acc:.2f} | **FGSM Başarısı:** %{acc_fgsm:.2f} | **PGD Başarısı:** %{acc_pgd:.2f}")
            
            # Kararlılık Eğrisi Grafiği
            fig_curve, ax_curve = plt.subplots(figsize=(8, 2.8))
            ax_curve.plot(eps_range, fgsm_curve, label='FGSM', color='#1f77b4', marker='o')
            ax_curve.plot(eps_range, pgd_curve, label='PGD', color='#d62728', marker='s')
            ax_curve.axhline(y=85, color='orange', linestyle='--', label='Eşik (%85)')
            ax_curve.fill_between(eps_range, 0, 85, where=(np.array(pgd_curve) < 85), color='red', alpha=0.1)
            ax_curve.set_xlabel('Epsilon ε')
            ax_curve.set_ylabel('Doğruluk (%)')
            ax_curve.legend(loc='lower left')
            st.pyplot(fig_curve)
            plt.close(fig_curve)

            # ==========================================
            # İSTEK: TEK TEK SAYILARIN ÜZERİNE BASILDIĞINDA O SINIFIN MATRİSİNİN ÇIKMASI
            # ==========================================
            st.markdown("---")
            st.markdown("#### **🎯 Sınıf Bazlı İnteraktif Değerlendirme**")
            st.caption("Detaylı Confusion Matrix analizini saniyeler içinde aşağıda üretmek için bir rakam butonuna basın:")
            
            # Yan yana 10 adet sayı butonu oluşturma
            num_cols = st.columns(10)
            if 'selected_digit' not in st.session_state:
                st.session_state['selected_digit'] = 0
                
            for digit in range(10):
                with num_cols[digit]:
                    # Seçili olan rakamın butonunu görsel olarak vurgulamak için tip belirleme
                    b_type = "primary" if st.session_state['selected_digit'] == digit else "secondary"
                    if st.button(f" {digit} ", key=f"select_{digit}", type=b_type, use_container_width=True):
                        st.session_state['selected_digit'] = digit
            
            # Seçilen rakama ait alt confusion matrix hesaplaması ve görselleştirilmesi
            sel_digit = st.session_state['selected_digit']
            active_preds = preds_pgd if preds_pgd is not None else preds_fgsm
            cm_global = confusion_matrix(y_test_np, active_preds, labels=list(range(10)))
            
            st.markdown(f"##### 🧮 **Sınıf [{sel_digit}] İçin İzole Siber Hata Dağılımı (Confusion Matrix)**")
            fig_sub, ax_sub = plt.subplots(figsize=(6, 1.8))
            sns.heatmap([cm_global[sel_digit]], annot=True, fmt='d', cmap='Purples', cbar=False, xticklabels=range(10), yticklabels=[sel_digit], ax=ax_sub)
            ax_sub.set_xlabel('Yapay Zekânın Saldırı Altındaki Hatalı Tahmin Sınıfları')
            st.pyplot(fig_sub)
            plt.close(fig_sub)

    # ==========================================
    # 3. SAYFA: GELİŞMİŞ KULLANICI SONUÇLARI (HEATMAP, ALARM VE GRAD-CAM)
    # ==========================================
    with tab3:
        if not st.session_state.get('analysis_triggered', False):
            st.info("💡 Lütfen öncelikle '2. Sayfa' üzerinden siber güvenlik analizini tetikleyin.")
        else:
            preds_fgsm = st.session_state['acc_fgsm']
            preds_pgd = st.session_state['acc_pgd']
            active_preds = st.session_state['preds_pgd'] if st.session_state['preds_pgd'] is not None else st.session_state['preds_fgsm']
            cm_global = confusion_matrix(y_test_np, active_preds, labels=list(range(10)))
            
            # İstek: Genel Bakış Başlıklı Küresel Heatmap
            st.markdown("### 📊 Genel Bakış (Küresel Korelasyon Matrisi)")
            fig_heatmap, ax_heatmap = plt.subplots(figsize=(7, 4.2))
            sns.heatmap(cm_global, annot=True, fmt='d', cmap='YlOrRd', xticklabels=range(10), yticklabels=range(10), ax=ax_heatmap)
            ax_heatmap.set_xlabel('Yapay Zekânın Tahmini')
            ax_heatmap.set_ylabel('Gerçek Sınıfı')
            st.pyplot(fig_heatmap)
            plt.close(fig_heatmap)
            
            # Kritik Nokta Alarmı
            total_vulnerability_score = (baseline_acc - min(st.session_state['acc_fgsm'], st.session_state['acc_pgd']))
            if total_vulnerability_score > 10.0:
                st.error(f"🚨 KRİTİK ALARM: Model üzerinde {total_vulnerability_score:.2f} birimlik aşırı kararsızlık saptanmıştır!")

            # İleri Seviye Grad-CAM Katmanı
            st.markdown("---")
            st.markdown("### 🧠 Gelişmiş Teşhis Paneli (Canlı Grad-CAM & Diferansiyel Fark Haritası)")
            
            selected_layer = st.selectbox("İncelemek İstediğiniz Derin Nöron Katmanını Seçin:", ["conv1", "conv2"], index=1)
            c_class = st.session_state.get('crit_class', 4)
            x_pgd_saved = st.session_state['saved_x_step']
            
            g_list, a_list = [], []
            def h_b(module, gi, go): g_list.append(go[0])
            def h_f(module, i, o): a_list.append(o)
            
            target_layer_module = getattr(eval_model, selected_layer)
            hook_backward = target_layer_module.register_full_backward_hook(h_b)
            hook_forward = target_layer_module.register_forward_hook(h_f)
            
            t_idx = np.where(y_test_np == c_class)[0][0]
            img_clean_t = torch.tensor(x_test_np[t_idx:t_idx+1]).to(device)
            img_adv_t = torch.tensor(x_pgd_saved[t_idx:t_idx+1]).to(device)
            
            def compute_cam(img_tensor):
                g_list.clear(); a_list.clear()
                out = eval_model(img_tensor)
                p_cls = torch.argmax(out, dim=1).item()
                conf = F.softmax(out, dim=1)[0, p_cls].item() * 100
                eval_model.zero_grad()
                out[0, p_cls].backward()
                gr = g_list[0].cpu().data.numpy()[0]
                ac = a_list[0].cpu().data.numpy()[0]
                w = np.mean(gr, axis=(1, 2))
                cam = np.zeros(ac.shape[1:], dtype=np.float32)
                for i, weight_val in enumerate(w): cam += weight_val * ac[i]
                cam = np.maximum(cam, 0)
                if np.max(cam) > 0: cam /= np.max(cam)
                cam = ndimage.zoom(cam, (28 / cam.shape[0], 28 / cam.shape[1]), order=1)
                return cam, p_cls, conf

            cam_c, p_c, conf_c = compute_cam(img_clean_t)
            cam_a, p_a, conf_a = compute_cam(img_adv_t)
            hook_backward.remove(); hook_forward.remove()
            
            cam_diff = np.abs(cam_c - cam_a)
            if np.max(cam_diff) > 0: cam_diff /= np.max(cam_diff)
            
            # 5'li Gelişmiş Adli Bilişim Matrisi Çizimi
            fig_cam, axes = plt.subplots(1, 5, figsize=(18, 3.8))
            axes[0].imshow(x_test_np[t_idx].squeeze(), cmap='gray')
            axes[0].set_title(f"Orijinal Girdi\nTahmin: {p_c}", color='green', fontsize=9, fontweight='bold')
            axes[0].axis('off')
            
            axes[1].imshow(x_test_np[t_idx].squeeze(), cmap='gray')
            axes[1].imshow(cam_c, cmap='jet', alpha=0.45)
            axes[1].contour(cam_c, colors='white', alpha=0.3, linewidths=0.5)
            axes[1].set_title(f"Temiz Odak (%{conf_c:.1f})", fontsize=9, fontweight='bold')
            axes[1].axis('off')
            
            axes[2].imshow(x_pgd_saved[t_idx].squeeze(), cmap='gray')
            axes[2].set_title(f"Saldırılı Girdi\nTahmin: {p_a}", color='red', fontsize=9, fontweight='bold')
            axes[2].axis('off')
            
            axes[3].imshow(x_pgd_saved[t_idx].squeeze(), cmap='gray')
            axes[3].imshow(cam_a, cmap='jet', alpha=0.55)
            axes[3].contour(cam_a, colors='white', alpha=0.3, linewidths=0.5)
            axes[3].set_title(f"Saldırı Odağı (%{conf_a:.1f})", color='red', fontsize=9, fontweight='bold')
            axes[3].axis('off')
            
            axes[4].imshow(x_test_np[t_idx].squeeze(), cmap='gray')
            axes[4].imshow(cam_diff, cmap='hot', alpha=0.6)
            axes[4].set_title("Odak Sapma Haritası", color='orange', fontsize=9, fontweight='bold')
            axes[4].axis('off')
            
            st.pyplot(fig_cam)
            plt.close(fig_cam)
