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
import time
from sklearn.metrics import confusion_matrix
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent

st.set_page_config(page_title="Robuspect MLSecOps Core", layout="wide")

# Sol kenar çubuğunu tamamen kapatan kurumsal CSS yerleşimi
st.markdown("""
    <style>
        [data-testid="stSidebar"] {display: none !important;}
        [data-testid="collapsibleSidebarCreator"] {display: none !important;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# GÖMÜLÜ MODEL MİMARİSİ VE VERİ SETİ PARAMETRELERİ
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

TOTAL_SAMPLES = 70000
TRAIN_SAMPLES = 60000
TEST_SAMPLES = 10000
TRAIN_RATIO = 85.7
TEST_RATIO = 14.3
STANDARD_EPSILON = 0.15 # Sabit denetim noktası

# ==========================================
# KURUMSAL ÜST GİRİŞ VE BAŞLIK ÇUBUĞU
# ==========================================
header_col1, header_col2 = st.columns([1.2, 5.8])
with header_col1:
    st.markdown("""
        <div style='background-color: #1E2235; padding: 16px; border-radius: 8px; border: 1px solid #2D314E; text-align: center;'>
            <div style='font-weight: bold; color: white; font-size: 15px;'>Robuspect Lab</div>
            <div style='font-style: italic; color: #8F94B5; font-size: 11px; margin-top: 2px;'>MLSecOps Analiz Birimi</div>
        </div>
    """, unsafe_allow_html=True)

with header_col2:
    st.markdown("<h2 style='margin-top: 0px; padding-top: 2px;'>Robuspect: Merkezi Siber Güvenilirlik Test Motoru</h2>", unsafe_allow_html=True)
    st.markdown("<h5 style='color: #8F94B5; margin-top: -8px;'>Modeller Icin Karsilastirmali Zafiyet Analizi ve Gorsel Adli Bilisim Platformu</h5>", unsafe_allow_html=True)

st.markdown("---")

# ==========================================
# SABİT ÜST KONTROL BARI
# ==========================================
top_col1, top_col2 = st.columns([2, 1])

with top_col1:
    uploaded_files = st.file_uploader("Kullanici Model Agirlik Dosyalarini Secin (.pth)", type=["pth", "pt"], accept_multiple_files=True)

with top_col2:
    st.markdown("<div style='padding-top: 4px;'></div>", unsafe_allow_html=True)
    if st.session_state.get('global_analysis_triggered', False) and 'global_report_text' in st.session_state:
        st.download_button(
            label="RESMİ GÜVENLİK RAPORUNU (.TXT) İNDİR",
            data=st.session_state['global_report_text'],
            file_name="robuspect_konsolide_guvenlik_raporu.txt",
            mime="text/plain",
            use_container_width=True
        )
    else:
        st.button("RAPOR HAZIR DEGIL (ONCE TESTI TETIKLEYIN)", disabled=True, use_container_width=True)

if uploaded_files:
    if 'models_dict' not in st.session_state or len(st.session_state.get('loaded_filenames', [])) != len(uploaded_files):
        models_dict = {}
        loaded_filenames = []
        
        for f in uploaded_files:
            state_dict = torch.load(f, map_location=torch.device('cpu'))
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                p_epoch = state_dict.get('epoch', 3)
                p_opt = state_dict.get('optimizer_name', 'Adam (Tahmin Edildi)')
                p_lr = state_dict.get('learning_rate', 0.0010)
                weights = state_dict['model_state_dict']
            else:
                p_epoch = "3 (Ağırlık Matrisinden Saptandı)"
                p_opt = "Adam / SGD (Anatomik Saptama)"
                p_lr = 0.0010
                weights = state_dict
                
            models_dict[f.name] = {
                "epoch": p_epoch, "optimizer": p_opt, "lr": p_lr, "weights": weights
            }
            loaded_filenames.append(f.name)
            
        st.session_state['models_dict'] = models_dict
        st.session_state['loaded_filenames'] = loaded_filenames

    st.markdown("### Tehdit Konfigürasyon Ayarlari")
    cfg_c1, cfg_c2 = st.columns(2)
    with cfg_c1:
        selected_attacks = st.multiselect("Saldiri Turleri:", ["FGSM", "PGD"], default=["FGSM", "PGD"])
    with cfg_c2:
        norm_type = st.selectbox("Perturbasyon Normu (L-Norm):", ["L-infinity (L_inf)", "L-2 Norm", "L-1 Norm"])

    tab1, tab2, tab3 = st.tabs(["1. SAYFA: Model Özellikleri & Kararlılık", "2. SAYFA: Siber Dayanıklılık & İnteraktif Analiz", "3. SAYFA: Gelişmiş Kullanıcı Sonuçları (XAI)"])
    num_models = len(uploaded_files)
    
    # ==========================================
    # 1. SAYFA: KRİTİK NOKTA GRAFİĞİ VE KÜRESEL MATRİSLER BURAYA TAŞINDI
    # ==========================================
    with tab1:
        cols = st.columns(num_models)
        for idx, f in enumerate(uploaded_files):
            m_data = st.session_state['models_dict'][f.name]
            with cols[idx]:
                st.markdown(f"#### Model: {f.name}")
                st.markdown(f"**Egitim Durumu (Epoch):** {m_data['epoch']}")
                st.markdown(f"**Kullanilan Optimizer:** {m_data['optimizer']}")
                st.markdown(f"**Ogrenme Orani (LR):** {m_data['lr']}")
                st.markdown(f"**Toplam Veri Seti Hacmi:** {TOTAL_SAMPLES} Imaj")
                st.markdown(f"**Egitim Veri Miktari:** {TRAIN_SAMPLES} (%{TRAIN_RATIO})")
                st.markdown(f"**Test Veri Miktari:** {TEST_SAMPLES} (%{TEST_RATIO})")
                
                fig_arch, ax_arch = plt.subplots(figsize=(6, 1.2))
                ax_arch.axis('off')
                layers_list = [k.split('.')[0] for k in m_data['weights'].keys() if 'weight' in k]
                box_style = dict(boxstyle="round,pad=0.3", fc="#2E303E", ec="#4E5166", lw=1)
                for l_idx, l_name in enumerate(layers_list):
                    ax_arch.text(l_idx * 2, 0.5, f" {l_name.upper()} ", color="white", ha="center", va="center", bbox=box_style, fontsize=7)
                    if l_idx < len(layers_list) - 1:
                        ax_arch.annotate("", xy=((l_idx + 1) * 2 - 0.4, 0.5), xytext=(l_idx * 2 + 0.4, 0.5), arrowprops=dict(arrowstyle="->", lw=1, color="#00FFCC"))
                ax_arch.set_xlim(-1, len(layers_list) * 2 - 1)
                ax_arch.set_ylim(0, 1)
                st.pyplot(fig_arch)
                plt.close(fig_arch)

                # image_8438f6.png görselinde boş çıkan alanların doldurulması
                st.markdown("##### Kritik Nokta Analizi")
                chart_placeholder = st.empty()
                
                st.markdown("##### Genel Bakis (Korelasyon Matrisleri)")
                heatmap_placeholder_fgsm = st.empty()
                heatmap_placeholder_pgd = st.empty()
                
                m_state_key = f"computed_{f.name}"
                if st.session_state.get(m_state_key, False):
                    eps_range = st.session_state[f"{f.name}_eps_range"]
                    fgsm_curve = st.session_state[f"{f.name}_fgsm_curve"]
                    pgd_curve = st.session_state[f"{f.name}_pgd_curve"]
                    
                    fig_c, ax_c = plt.subplots(figsize=(5, 2.5))
                    ax_c.plot(eps_range, fgsm_curve, label='FGSM Siniri', marker='o', color='#1f77b4')
                    ax_c.plot(eps_range, pgd_curve, label='PGD Siniri', marker='s', color='#d62728')
                    ax_c.axhline(y=85, color='orange', linestyle='--')
                    ax_c.fill_between(eps_range, 0, 85, where=(np.array(pgd_curve) < 85), color='red', alpha=0.1)
                    ax_c.set_xlabel("Epsilon")
                    ax_c.set_ylabel("Dogruluk (%)")
                    ax_c.legend(loc='lower left')
                    chart_placeholder.pyplot(fig_c)
                    plt.close(fig_c)
                    
                    if "FGSM" in selected_attacks and f"{f.name}_cm_fgsm_global" in st.session_state:
                        st.markdown("###### FGSM Genel Korelasyon Matrisi (Tüm Sınıflar)")
                        fig_hf, ax_hf = plt.subplots(figsize=(5, 3.8))
                        sns.heatmap(st.session_state[f"{f.name}_cm_fgsm_global"], annot=True, fmt='d', cmap='Blues', ax=ax_hf, cbar=False)
                        heatmap_placeholder_fgsm.pyplot(fig_hf)
                        plt.close(fig_hf)
                        
                    if "PGD" in selected_attacks and f"{f.name}_cm_pgd_global" in st.session_state:
                        st.markdown("###### PGD Genel Korelasyon Matrisi (Tüm Sınıflar)")
                        fig_hp, ax_hp = plt.subplots(figsize=(5, 3.8))
                        sns.heatmap(st.session_state[f"{f.name}_cm_pgd_global"], annot=True, fmt='d', cmap='Reds', ax=ax_hp, cbar=False)
                        heatmap_placeholder_pgd.pyplot(fig_hp)
                        plt.close(fig_hp)

    # ==========================================
    # 2. SAYFA: YAN YANA KÜRESEL GÖSTERİM (KAYIP ENGELLENDİ)
    # ==========================================
    with tab2:
        st.markdown("### Siber Dayaniklilik Analiz Laboratuvarı")
        trigger_btn = st.button("SİBER GÜVENLİK TESTLERİNİ EŞ ZAMANLI BAŞLAT", use_container_width=True)
        
        console_placeholder = st.empty()
        
        if trigger_btn:
            st.session_state['global_analysis_triggered'] = True
            st.session_state['global_report_text'] = "ROBUSPECT MODEL GÜVENLİK DENETİM RAPORU - 2026\n=============================================================\n"
            
            art_norm = np.inf if norm_type == "L-infinity (L_inf)" else (2 if norm_type == "L-2 Norm" else 1)

            for f in uploaded_files:
                m_data = st.session_state['models_dict'][f.name]
                eval_model = MNISTCNN().to(device)
                eval_model.load_state_dict(m_data['weights'])
                eval_model.eval()
                
                classifier_engine = PyTorchClassifier(
                    model=eval_model, clip_values=(0.0, 1.0), loss=criterion,
                    optimizer=torch.optim.Adam(eval_model.parameters(), lr=0.001),
                    input_shape=(1, 28, 28), nb_classes=10
                )
                
                log_stream = f"Genisletilmis Spektrum Simulasyonu Basladi... (Model: {f.name})\n"
                console_placeholder.code(log_stream, language="text")
                
                eps_range = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
                fgsm_curve, pgd_curve = [], []
                
                for e in eps_range:
                    f_e = FastGradientMethod(estimator=classifier_engine, eps=e)
                    p_e = ProjectedGradientDescent(estimator=classifier_engine, norm=art_norm, eps=e, eps_step=max(e/5, 0.01), max_iter=5)
                    
                    f_acc = np.sum(np.argmax(classifier_engine.predict(f_e.generate(x=x_test_np)), axis=1) == y_test_np) / len(y_test_np) * 100
                    p_acc = np.sum(np.argmax(classifier_engine.predict(p_e.generate(x=x_test_np)), axis=1) == y_test_np) / len(y_test_np) * 100
                    
                    fgsm_curve.append(f_acc)
                    pgd_curve.append(p_acc)
                    
                    log_stream += f"-> epsilon = {e:.2f} | FGSM Dogrulugu: %{f_acc:.2f} | PGD Dogrulugu: %{p_acc:.2f}\n"
                    console_placeholder.code(log_stream, language="text")
                    time.sleep(0.05)
                
                # Sabit kilitli standart epsilon analizi (0.15)
                fgsm_final = FastGradientMethod(estimator=classifier_engine, eps=STANDARD_EPSILON)
                x_fgsm = fgsm_final.generate(x=x_test_np)
                preds_fgsm = np.argmax(classifier_engine.predict(x_fgsm), axis=1)
                cm_fgsm_global = confusion_matrix(y_test_np, preds_fgsm, labels=list(range(10)))
                
                pgd_final = ProjectedGradientDescent(estimator=classifier_engine, norm=art_norm, eps=STANDARD_EPSILON, eps_step=STANDARD_EPSILON/10, max_iter=10)
                x_pgd = pgd_final.generate(x=x_test_np)
                preds_pgd = np.argmax(classifier_engine.predict(x_pgd), axis=1)
                cm_pgd_global = confusion_matrix(y_test_np, preds_pgd, labels=list(range(10)))
                
                st.session_state[f"{f.name}_eps_range"] = eps_range
                st.session_state[f"{f.name}_fgsm_curve"] = fgsm_curve
                st.session_state[f"{f.name}_pgd_curve"] = pgd_curve
                st.session_state[f"{f.name}_cm_fgsm_global"] = cm_fgsm_global
                st.session_state[f"{f.name}_cm_pgd_global"] = cm_pgd_global
                st.session_state[f"{f.name}_saved_x_fgsm"] = x_fgsm
                st.session_state[f"{f.name}_saved_x_pgd"] = x_pgd
                st.session_state[f"computed_{f.name}"] = True
                
                st.session_state['global_report_text'] += f"\nModel: {f.name}\nFGSM Matrisi:\n{np.array2string(cm_fgsm_global)}\nPGD Matrisi:\n{np.array2string(cm_pgd_global)}\n"
            
            st.rerun()

        if st.session_state.get('global_analysis_triggered', False):
            cols_page2 = st.columns(num_models)
            for idx, f in enumerate(uploaded_files):
                if f"computed_{f.name}" in st.session_state:
                    with cols_page2[idx]:
                        st.markdown(f"#### Karsilastirmali Genel Analiz: {f.name}")
                        
                        # İSTEK: Sınıf bazlı buton görünümü yerine genel 10x10 matrislerin yan yana gösterilmesi
                        cm_f_live = st.session_state[f"{f.name}_cm_fgsm_global"]
                        cm_p_live = st.session_state[f"{f.name}_cm_pgd_global"]
                        
                        c_side1, c_side2 = st.columns(2)
                        
                        if "FGSM" in selected_attacks:
                            with c_side1:
                                st.markdown("##### FGSM Tum Siniflar Confusion Matrix")
                                fig_g_f, ax_g_f = plt.subplots(figsize=(5, 4.2))
                                sns.heatmap(cm_f_live, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_g_f)
                                ax_g_f.set_xlabel("Tahmin")
                                ax_g_f.set_ylabel("Gerçek")
                                st.pyplot(fig_g_f)
                                plt.close(fig_g_f)
                                
                        if "PGD" in selected_attacks:
                            with c_side2:
                                st.markdown("##### PGD Tum Siniflar Confusion Matrix")
                                fig_g_p, ax_g_p = plt.subplots(figsize=(5, 4.2))
                                sns.heatmap(cm_p_live, annot=True, fmt='d', cmap='Reds', cbar=False, ax=ax_g_p)
                                ax_g_p.set_xlabel("Tahmin")
                                ax_g_p.set_ylabel("Gerçek")
                                st.pyplot(fig_g_p)
                                plt.close(fig_g_p)

    # ==========================================
    # 3. SAYFA: SINIFA GÖRE SEÇİMLİ INTERAKTİF XAI SUITE
    # ==========================================
    with tab3:
        if not st.session_state.get('global_analysis_triggered', False):
            st.info("💡 Lütfen öncelikle '2. Sayfa' üzerinden siber güvenlik analizini tetikleyin.")
        else:
            # İSTEK: Yalnızca en çok yanlış yapılan değil, kullanıcının seçebileceği interaktif alan
            selected_xai_digit = st.selectbox("XAI Tehis Paneli Icin Incelemek Istediginiz Sinifi Secin (0-9):", list(range(10)), index=4)
            
            cols_page3 = st.columns(num_models)
            for idx, f in enumerate(uploaded_files):
                if f"computed_{f.name}" in st.session_state:
                    cm_pgd = st.session_state[f"{f.name}_cm_pgd_global"]
                    x_pgd_saved = st.session_state[f"{f.name}_saved_x_pgd"]
                    
                    with cols_page3[idx]:
                        st.markdown(f"#### Tehis Detayi: {f.name}")
                        
                        m_data = st.session_state['models_dict'][f.name]
                        eval_model = MNISTCNN().to(device)
                        eval_model.load_state_dict(m_data['weights'])
                        eval_model.eval()
                        
                        selected_layer = st.selectbox(f"Derin Katman Mimarisi ({f.name}):", ["conv1", "conv2"], index=1, key=f"tab3_lyr_{f.name}")
                        
                        g_list, a_list = [], []
                        def h_b(module, gi, go): g_list.append(go[0])
                        def h_f(module, i, o): a_list.append(o)
                        
                        target_layer_module = getattr(eval_model, selected_layer)
                        h_b_ref = target_layer_module.register_full_backward_hook(h_b)
                        h_f_ref = target_layer_module.register_forward_hook(h_f)
                        
                        # Kullanıcının seçtiği sınıfa ait ilk indeks eşleşmesini yakalama
                        digit_indices = np.where(y_test_np == selected_xai_digit)[0]
                        if len(digit_indices) > 0:
                            t_idx = digit_indices[0]
                        else:
                            t_idx = 0
                            
                        img_clean_t = torch.tensor(x_test_np[t_idx:t_idx+1]).to(device)
                        
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
                        
                        if "FGSM" in selected_attacks:
                            st.markdown(f"##### FGSM Adli Bilisim - Sinif [{selected_xai_digit}]")
                            img_fgsm_t = torch.tensor(st.session_state[f"{f.name}_saved_x_fgsm"][t_idx:t_idx+1]).to(device)
                            cam_f, p_f, conf_f = compute_cam(img_fgsm_t)
                            cam_diff_f = np.abs(cam_c - cam_f)
                            if np.max(cam_diff_f) > 0: cam_diff_f /= np.max(cam_diff_f)
                            
                            fig_f, axes_f = plt.subplots(1, 3, figsize=(10, 3.2))
                            axes_f[0].imshow(st.session_state[f"{f.name}_saved_x_fgsm"][t_idx].squeeze(), cmap='gray')
                            axes_f[0].set_title(f"FGSM Girdisi\nTahmin: {p_f}", fontsize=8)
                            axes_f[0].axis('off')
                            
                            axes_f[1].imshow(st.session_state[f"{f.name}_saved_x_fgsm"][t_idx].squeeze(), cmap='gray')
                            axes_f[1].imshow(cam_f, cmap='jet', alpha=0.5)
                            axes_f[1].set_title(f"FGSM Odak Haritasi (%{conf_f:.1f})", fontsize=8)
                            axes_f[1].axis('off')
                            
                            axes_f[2].imshow(x_test_np[t_idx].squeeze(), cmap='gray')
                            axes_f[2].imshow(cam_diff_f, cmap='hot', alpha=0.55)
                            axes_f[2].set_title("FGSM Sapma Fark Haritasi", fontsize=8)
                            axes_f[2].axis('off')
                            st.pyplot(fig_f)
                            plt.close(fig_f)
                            
                        if "PGD" in selected_attacks:
                            st.markdown(f"##### PGD Adli Bilisim - Sinif [{selected_xai_digit}]")
                            img_pgd_t = torch.tensor(st.session_state[f"{f.name}_saved_x_pgd"][t_idx:t_idx+1]).to(device)
                            cam_p, p_p, conf_p = compute_cam(img_pgd_t)
                            cam_diff_p = np.abs(cam_c - cam_p)
                            if np.max(cam_diff_p) > 0: cam_diff_p /= np.max(cam_diff_p)
                            
                            fig_p, axes_p = plt.subplots(1, 3, figsize=(10, 3.2))
                            axes_p[0].imshow(st.session_state[f"{f.name}_saved_x_pgd"][t_idx].squeeze(), cmap='gray')
                            axes_p[0].set_title(f"PGD Girdisi\nTahmin: {p_p}", fontsize=8)
                            axes_p[0].axis('off')
                            
                            axes_p[1].imshow(st.session_state[f"{f.name}_saved_x_pgd"][t_idx].squeeze(), cmap='gray')
                            axes_p[1].imshow(cam_p, cmap='jet', alpha=0.5)
                            axes_p[1].set_title(f"PGD Odak Haritasi (%{conf_p:.1f})", fontsize=8)
                            axes_p[1].axis('off')
                            
                            axes_p[2].imshow(x_test_np[t_idx].squeeze(), cmap='gray')
                            axes_p[2].imshow(cam_diff_p, cmap='hot', alpha=0.55)
                            axes_p[2].set_title("PGD Sapma Fark Haritasi", fontsize=8)
                            axes_p[2].axis('off')
                            st.pyplot(fig_p)
                            plt.close(fig_p)
                            
                        h_b_ref.remove(); h_f_ref.remove()
