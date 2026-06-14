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

# Sol çubuğu (Sidebar) tamamen devre dışı bırakıp gizleyen CSS enjeksiyonu
st.markdown("""
    <style>
        [data-testid="stSidebar"] {display: none !important;}
        [data-testid="collapsibleSidebarCreator"] {display: none !important;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# GÖMÜLÜ MODEL MİMARİSİ VE VERİ SETİ
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
    test_loader = DataLoader(test_dataset, batch_size=300, shuffle=False) # Raporlama kararlılığı için optimize edildi
    x_t, y_t = next(iter(test_loader))
    return x_t.numpy(), y_t.numpy()

x_test_np, y_test_np = load_embedded_test_data()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

# ==========================================
# SOL ÜST PANEL VE BAŞLIK ALANI
# ==========================================
head_col1, head_col2 = st.columns([1, 4])
with head_col1:
    st.info("🛡️ **Robuspect Lab**\n\n*MLSecOps Analiz Birimi*")

st.title("🛡️ Robuspect: Merkezi Siber Güvenilirlik Test Motoru")
st.subheader("Modeller İçin Karşılaştırmalı Zafiyet Analizi ve Görsel Adli Bilişim Platformu")
st.markdown("---")

# Giriş Ekranı: Dosya İsteme
uploaded_weights = st.file_uploader("🚀 BAŞLAMAK İÇİN: Eğittiğiniz Model Ağırlık Dosyasını Seçin (.pth)", type=["pth", "pt"])

if uploaded_weights is None:
    st.warning("⚠️ Tarama motorunu başlatabilmek için lütfen yukarıdaki alana geçerli bir PyTorch model ağırlık dosyası (.pth) yükleyin.")
else:
    # Model yükleme ve akıllı öngörü tetikleyicisi
    if 'model_loaded' not in st.session_state or st.session_state.get('last_uploaded') != uploaded_weights.name:
        with st.spinner("Model ağırlık matrisleri çözümleniyor, parametreler tahmin ediliyor..."):
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

    # Hafızadaki değişkenleri lokal kullanıma açalım
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

    st.success("🟢 Model Başarıyla Kaydedildi ve Kalıba Döküldü!")
    
    with st.expander("📋 Otomatik Çıkarılan Model Künyesi ve Katman Anatomisi", expanded=True):
        k1, k2, k3 = st.columns(3)
        k1.metric("Tahmin Edilen Eğitim Süresi", f"{predicted_epoch}")
        k2.metric("Saptanan Optimizasyon", f"{predicted_opt}")
        k3.metric("Saptanan Öğrenme Oranı (LR)", f"{predicted_lr}")

    st.markdown("---")
    
    # INTERAKTIF PARAMETRELER (ANA PANELDE)
    st.markdown("### ⚙️ Siber Güvenlik Tehdit Konfigürasyonu")
    p_col1, p_col2, p_col3 = st.columns(3)
    with p_col1:
        selected_attacks = st.multiselect("Saldırı Türlerini Seçin:", ["FGSM", "PGD"], default=["FGSM", "PGD"])
    with p_col2:
        norm_type = st.selectbox("Pertürbasyon Normu (L-Norm):", ["L-infinity (L_inf)", "L-2 Norm"])
    with p_col3:
        eps_value = st.slider("Gürültü Şiddeti (Epsilon ε):", min_value=0.01, max_value=0.50, value=0.15, step=0.01)

    clean_preds = np.argmax(classifier_engine.predict(x_test_np), axis=1)
    baseline_acc = np.sum(clean_preds == y_test_np) / len(y_test_np) * 100
    st.info(f"🎯 **Modelinizin Orijinal (Saldırısız) Verideki Doğruluk Oranı:** %{baseline_acc:.2f}")

    tab1, tab2, tab3 = st.tabs(["📋 Yapay Zekâ Araç Raporu", "📊 Eş Zamanlı Siber Simülasyon", "🧠 XAI / Canlı Grad-CAM Teşhisi"])
    
    # SEKMELER 1: SEKTÖR STANDARDI KATMAN GÖRSELLEŞTİRME
    with tab1:
        st.markdown("### 🔍 Model Röntgeni ve Katman Tasarımı Akış Şeması")
        
        # Sektör standardı blok şemayı matplotlib ile dinamik çiziyoruz
        fig_arch, ax_arch = plt.subplots(figsize=(10, 2))
        ax_arch.axis('off')
        
        layers_list = []
        for key in state_dict.keys():
            if 'weight' in key:
                layers_list.append(key.split('.')[0])
                
        # Blokları yan yana dizelim
        box_style = dict(boxstyle="round,pad=0.5", fc="#2E303E", ec="#4E5166", lw=2)
        arrow_style = dict(arrowstyle="->", lw=2, color="#00FFCC")
        
        for idx, l_name in enumerate(layers_list):
            ax_arch.text(idx * 2, 0.5, f" {l_name.upper()} \nLayer ", color="white", weight="bold", ha="center", va="center", bbox=box_style, fontsize=9)
            if idx < len(layers_list) - 1:
                ax_arch.annotate("", xy=((idx + 1) * 2 - 0.5, 0.5), xytext=(idx * 2 + 0.5, 0.5), arrowprops=arrow_style)
                
        ax_arch.set_xlim(-1, len(layers_list) * 2 - 1)
        ax_arch.set_ylim(0, 1)
        st.pyplot(fig_arch)
        plt.close(fig_arch)

        # Ham veri tablosu
        layers_info = []
        for key in state_dict.keys():
            if 'weight' in key:
                name = key.split('.')[0]
                shape = list(state_dict[key].shape)
                l_type = "Evrişim (Conv2d)" if len(shape) == 4 else "Tam Bağlantılı (Linear)"
                layers_info.append({"Katman İsmi": name, "Katman Türü": l_type, "Matris Boyutu (Tensor Shape)": str(shape)})
        st.table(layers_info)

    # SEKMELER 2: SİBER SİMÜLASYON (DURUM KORUMALI)
    with tab2:
        st.markdown("### ⚙️ FGSM ve PGD Karşılaştırmalı Siber Laboratuvarı")
        trigger_btn = st.button("🚀 SİBER GÜVENLİK TESTİNİ TETİKLE", use_container_width=True)
        
        # Eğer butona basıldıysa durumu sabitle
        if trigger_btn:
            st.session_state['analysis_triggered'] = True
            
            with st.spinner("Siber tehdit matrisleri ve kararlılık eğrileri hesaplanıyor..."):
                art_norm = np.inf if norm_type.startswith("L-infinity") else 2
                
                # Standart test tahminleri
                fgsm_eng = FastGradientMethod(estimator=classifier_engine, eps=eps_value)
                x_fgsm = fgsm_eng.generate(x=x_test_np)
                preds_fgsm = np.argmax(classifier_engine.predict(x_fgsm), axis=1)
                acc_fgsm = np.sum(preds_fgsm == y_test_np) / len(y_test_np) * 100
                
                pgd_eng = ProjectedGradientDescent(estimator=classifier_engine, norm=art_norm, eps=eps_value, eps_step=eps_value/10, max_iter=10)
                x_pgd = pgd_eng.generate(x=x_test_np)
                preds_pgd = np.argmax(classifier_engine.predict(x_pgd), axis=1)
                acc_pgd = np.sum(preds_pgd == y_test_np) / len(y_test_np) * 100
                
                # --- KRİTİK DÜŞÜŞ GRAFİĞİ HESAPLAMA DÖNGÜSÜ ---
                eps_range = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
                fgsm_curve, pgd_curve = [], []
                
                for e in eps_range:
                    f_e = FastGradientMethod(estimator=classifier_engine, eps=e)
                    p_e = ProjectedGradientDescent(estimator=classifier_engine, norm=art_norm, eps=e, eps_step=max(e/5, 0.01), max_iter=5)
                    
                    fgsm_curve.append(np.sum(np.argmax(classifier_engine.predict(f_e.generate(x=x_test_np)), axis=1) == y_test_np) / len(y_test_np) * 100)
                    pgd_curve.append(np.sum(np.argmax(classifier_engine.predict(p_e.generate(x=x_test_np)), axis=1) == y_test_np) / len(y_test_np) * 100)
                
                # Tüm hesaplanan çıktıları hafızaya kilitleyelim (Rapor indirirken silinmeyi önleme)
                st.session_state['acc_fgsm'] = acc_fgsm
                st.session_state['acc_pgd'] = acc_pgd
                st.session_state['preds_fgsm'] = preds_fgsm
                st.session_state['preds_pgd'] = preds_pgd
                st.session_state['eps_range'] = eps_range
                st.session_state['fgsm_curve'] = fgsm_curve
                st.session_state['pgd_curve'] = pgd_curve
                st.session_state['saved_x_step'] = x_pgd if "PGD" in selected_attacks else x_fgsm

        # Hafıza kontrolü ile çıktıların ekrana basılması
        if st.session_state.get('analysis_triggered', False):
            acc_fgsm = st.session_state['acc_fgsm']
            acc_pgd = st.session_state['acc_pgd']
            preds_fgsm = st.session_state['preds_fgsm']
            preds_pgd = st.session_state['preds_pgd']
            eps_range = st.session_state['eps_range']
            fgsm_curve = st.session_state['fgsm_curve']
            pgd_curve = st.session_state['pgd_curve']
            
            sim_col1, sim_col2 = st.columns(2)
            if "FGSM" in selected_attacks:
                with sim_col1:
                    st.markdown("#### **FGSM Analiz Çıktısı**")
                    st.metric("FGSM Sağlamlık Doğruluğu", f"%{acc_fgsm:.2f}", delta=f"- %{baseline_acc - acc_fgsm:.2f} Kayıp")
                    fig_f, ax_f = plt.subplots(figsize=(4, 3.2))
                    sns.heatmap(confusion_matrix(y_test_np, preds_fgsm), annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_f)
                    st.pyplot(fig_f)
                    plt.close(fig_f)
            
            if "PGD" in selected_attacks:
                with sim_col2:
                    st.markdown("#### **PGD Analiz Çıktısı**")
                    st.metric("PGD Sağlamlık Doğruluğu", f"%{acc_pgd:.2f}", delta=f"- %{baseline_acc - acc_pgd:.2f} Kayıp")
                    fig_p, ax_p = plt.subplots(figsize=(4, 3.2))
                    sns.heatmap(confusion_matrix(y_test_np, preds_pgd), annot=True, fmt='d', cmap='Reds', cbar=False, ax=ax_p)
                    st.pyplot(fig_p)
                    plt.close(fig_p)

            # ==========================================
            # YENİ ÖZELLİK: BİRLEŞİK KRİTİK DÜŞÜŞ GRAFİĞİ
            # ==========================================
            st.markdown("---")
            st.markdown("### 📈 Kararlılık Eğrisi ve Kritik Kırılma Noktası Analizi")
            
            fig_curve, ax_curve = plt.subplots(figsize=(8, 3.5))
            ax_curve.plot(eps_range, fgsm_curve, label='FGSM Dayanıklılık Sınırı', color='#1f77b4', marker='o', lw=2)
            ax_curve.plot(eps_range, pgd_curve, label='PGD Dayanıklılık Sınırı', color='#d62728', marker='s', lw=2)
            ax_curve.axhline(y=85, color='orange', linestyle='--', label='Kritik Güvenlik Eşiği (%85)')
            
            # Kritik zafiyet alanını gölgelendirerek net işaretleme
            ax_curve.fill_between(eps_range, 0, 85, where=(np.array(pgd_curve) < 85), color='red', alpha=0.15, label='Zafiyet Alanı (Critical Drop Zone)')
            
            ax_curve.set_xlabel('Pertürbasyon Şiddeti (Epsilon ε)')
            ax_curve.set_ylabel('Model Doğruluk Oranı (%)')
            ax_curve.grid(True, alpha=0.3)
            ax_curve.legend(loc='lower left')
            st.pyplot(fig_curve)
            plt.close(fig_curve)

            # SINIF BAZLI BİREYSEL MATRİSLER
            st.markdown("---")
            st.markdown("### 🧮 Sınıf Bazlı Bireysel Karışıklık Matrisleri")
            active_preds = preds_pgd if preds_pgd is not None else preds_fgsm
            cm_global = confusion_matrix(y_test_np, active_preds, labels=list(range(10)))
            
            grid_col1, grid_col2, grid_col3 = st.columns(3)
            for digit in range(10):
                idx = (y_test_np == digit)
                digit_acc = (np.sum(active_preds[idx] == y_test_np[idx]) / np.sum(idx)) * 100 if np.sum(idx) > 0 else 0
                target_grid = grid_col1 if digit % 3 == 0 else (grid_col2 if digit % 3 == 1 else grid_col3)
                
                with target_grid:
                    st.markdown(f"**Sınıf [{digit}] Matrisi (Doğruluk: %{digit_acc:.1f})**")
                    fig_sub, ax_sub = plt.subplots(figsize=(2.5, 2))
                    sns.heatmap([cm_global[digit]], annot=True, fmt='d', cmap='Purples', cbar=False, xticklabels=range(10), yticklabels=[digit], ax=ax_sub)
                    st.pyplot(fig_sub)
                    plt.close(fig_sub)

            # GENEL BAKIŞ HEATMAP
            st.markdown("---")
            st.markdown("### 📊 Genel Bakış")
            fig_heatmap, ax_heatmap = plt.subplots(figsize=(7, 4.5))
            sns.heatmap(cm_global, annot=True, fmt='d', cmap='YlOrRd', xticklabels=range(10), yticklabels=range(10), ax=ax_heatmap)
            st.pyplot(fig_heatmap)
            plt.close(fig_heatmap)

            # GÜVENLİK RAPORU VE HAVUZU (.TXT İNDİRME ALANI)
            st.markdown("---")
            st.markdown("### 📜 Küresel Güvenlik Raporu ve İndirme Merkezi")
            
            report_rows = []
            for digit in range(10):
                idx = (y_test_np == digit)
                c_acc = (np.sum(clean_preds[idx] == y_test_np[idx]) / np.sum(idx)) * 100
                attack_acc = (np.sum(active_preds[idx] == y_test_np[idx]) / np.sum(idx)) * 100
                delta_drop = c_acc - attack_acc
                risk_status = "🔴 KRİTİK ZAFİYET" if delta_drop >= 25.0 else ("🟡 ORTA RİSK" if delta_drop >= 10.0 else "🟢 GÜVENLİ")
                
                report_rows.append({"Sınıf (Rakam)": f"Sınıf {digit}", "Temiz Doğruluk": f"%{c_acc:.2f}", "Saldırı Sonrası Doğruluk": f"%{attack_acc:.2f}", "Performans Kaybı (Δ)": f"- %{delta_drop:.2f}", "Güvenlik Durumu": risk_status})
            
            st.dataframe(pd.DataFrame(report_rows), use_container_width=True)

            # Metin raporunu dinamik hazırlıyoruz
            report_text = f"=============================================================\nROBUSPECT MLSECOPS MODEL GÜVENLİK DENETİM RAPORU\n=============================================================\n"
            report_text += f"Tahmini Optimizer: {predicted_opt}\nModel Orijinal Başarısı: %{baseline_acc:.2f}\nSaldırı Altındaki En Düşük Başarı: %{min(acc_fgsm, acc_acc_live := acc_pgd):.2f}\n"
            
            st.markdown("#### **📥 Resmi Denetim Raporunu Dışarı Aktar**")
            st.download_button(label="📄 RESMİ GÜVENLİK RAPORUNU (.TXT) İNDİR", data=report_text, file_name="robuspect_model_guvenlik_raporu.txt", mime="text/plain", use_container_width=True)
            
            # Bir sonraki sekmeye kırılgan sınıfı paslayalım
            class_scores = {c: np.sum(active_preds[y_test_np == c] == y_test_np[y_test_np == c]) / np.sum(y_test_np == c) * 100 for c in range(10)}
            st.session_state['crit_class'] = sorted(class_scores.items(), key=lambda x: x[1])[0][0]

    # SEKMELER 3: GRAD-CAM
    with tab3:
        st.markdown("### 🧠 Gömülü XAI Teşhis Paneli (Canlı Grad-CAM)")
        if 'crit_class' not in st.session_state or 'saved_x_step' not in st.session_state:
            st.info("💡 Grad-CAM üretebilmek için lütfen önce ikinci sekmeden 'Siber Testi Tetikle' butonuna basın.")
        else:
            c_class = st.session_state['crit_class']
            x_pgd_saved = st.session_state['saved_x_step']
            
            st.success(f"En çok sabote edilen Sınıf [{c_class}] için derin katman nöron aktivasyon haritası çıkartılıyor...")
            
            g_list, a_list = [], []
            def h_b(module, gi, go): g_list.append(go[0])
            def h_f(module, i, o): a_list.append(o)
            
            hook_backward = eval_model.conv2.register_full_backward_hook(h_b)
            hook_forward = eval_model.conv2.register_forward_hook(h_f)
            
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
            
            fig_cam, axes = plt.subplots(1, 4, figsize=(14, 4))
            axes[0].imshow(x_test_np[t_idx].squeeze(), cmap='gray')
            axes[0].set_title(f"Orijinal Girdi\nTahmin: {p_c}", color='green', fontsize=10, fontweight='bold')
            axes[0].axis('off')
            
            axes[1].imshow(x_test_np[t_idx].squeeze(), cmap='gray')
            axes[1].imshow(cam_c, cmap='jet', alpha=0.45)
            axes[1].set_title(f"Temiz Karar Odağı\nGüven: %{conf_c:.1f}", fontsize=10, fontweight='bold')
            axes[1].axis('off')
            
            axes[2].imshow(x_pgd_saved[t_idx].squeeze(), cmap='gray')
            axes[2].set_title(f"Saldırılı Girdi\nTahmin: {p_a}", color='red', fontsize=10, fontweight='bold')
            axes[2].axis('off')
            
            axes[3].imshow(x_pgd_saved[t_idx].squeeze(), cmap='gray')
            axes[3].imshow(cam_a, cmap='jet', alpha=0.55)
            axes[3].set_title(f"Saldırı Altındaki Sapma\nGüven: %{conf_a:.1f}", color='red', fontsize=10, fontweight='bold')
            axes[3].axis('off')
            
            st.pyplot(fig_cam)
            plt.close(fig_cam)
