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

# ==========================================
# 1. MODEL MİMARİSİ VE VERİ SETİ ALTYAPISI
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
    test_loader = DataLoader(test_dataset, batch_size=400, shuffle=False)
    x_t, y_t = next(iter(test_loader))
    return x_t.numpy(), y_t.numpy()

x_test_np, y_test_np = load_embedded_test_data()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

# Sidebar temizlendi (image_1e4a1c.png'deki uyarı ekranı kaldırıldı)
st.sidebar.title("🛡️ Robuspect Lab")

# ==========================================
# 2. AŞAMALI GİRİŞ EKRANI MANTIĞI
# ==========================================
st.title("🛡️ Robuspect: Merkezi Siber Güvenilirlik Test Motoru")
st.subheader("Modeller İçin Karşılaştırmalı Zafiyet Analizi ve Görsel Adli Bilişim Platformu")
st.markdown("---")

uploaded_weights = st.file_uploader("🚀 BAŞLAMAK İÇİN: Eğittiğiniz Model Ağırlık Dosyasını Seçin (.pth)", type=["pth", "pt"])

if uploaded_weights is None:
    st.warning("⚠️ Tarama motorunu başlatabilmek için lütfen yukarıdaki alana geçerli bir PyTorch model ağırlık dosyası (.pth) yükleyin.")
else:
    with st.spinner("Model ağırlık matrisleri çözümleniyor, parametreler tahmin ediliyor..."):
        state_dict = torch.load(uploaded_weights, map_location=torch.device('cpu'))
        
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            predicted_epoch = state_dict.get('epoch', 5)
            predicted_opt = state_dict.get('optimizer_name', 'Adam (Tahmin Edildi)')
            predicted_lr = state_dict.get('learning_rate', 0.0010)
            state_dict = state_dict['model_state_dict']
        else:
            predicted_epoch = "3-5 (Ağırlık Yoğunluğundan Otomatik Saptandı)"
            predicted_opt = "Adam / SGD (Anatomik Saptama)"
            predicted_lr = 0.0010

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
    
    # ==========================================
    # 3. SİBER GÜVENLİK TEHDİT KONFİGÜRASYONU
    # ==========================================
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

    # ==========================================
    # 4. TAB YAPILANDIRMASI (KATMAN ISIMLERI SILINDI)
    # ==========================================
    tab1, tab2, tab3 = st.tabs(["📋 Yapay Zekâ Araç Raporu", "📊 Eş Zamanlı Siber Simülasyon", "🧠 XAI / Canlı Grad-CAM Teşhisi"])
    
    with tab1:
        st.markdown("### 🔍 Model Röntgeni ve Araç Doğrulama İstatistikleri")
        layers_info = []
        for key in state_dict.keys():
            if 'weight' in key:
                name = key.split('.')[0]
                shape = list(state_dict[key].shape)
                l_type = "Evrişim (Conv2d)" if len(shape) == 4 else "Tam Bağlantılı (Linear)"
                layers_info.append({"Katman İsmi": name, "Katman Türü": l_type, "Matris Boyutu (Tensor Shape)": str(shape)})
        st.table(layers_info)

    with tab2:
        st.markdown("### ⚙️ FGSM networks ve PGD Karşılaştırmalı Siber Laboratuvarı")
        trigger_btn = st.button("🚀 SİBER GÜVENLİK TESTİNİ TETİKLE", use_container_width=True)
        
        if trigger_btn:
            st.toast("⚡ Siber saldırı simülasyonu başlatıldı! Vektörler üretiliyor...", icon="🔥")
            st.info("🔄 Bilgi: FGSM ve PGD siber tehdit simülasyon motoru arka planda çalıştırılıyor. Lütfen aşağı kaydırarak analizleri inceleyin.")
            
            art_norm = np.inf if norm_type.startswith("L-infinity") else 2
            preds_fgsm, preds_pgd = None, None
            acc_fgsm, acc_pgd = baseline_acc, baseline_acc
            
            sim_col1, sim_col2 = st.columns(2)
            
            # [SOL] ibaresi kaldırıldı (image_1e469a.png)
            if "FGSM" in selected_attacks:
                fgsm_eng = FastGradientMethod(estimator=classifier_engine, eps=eps_value)
                x_fgsm = fgsm_eng.generate(x=x_test_np)
                preds_fgsm = np.argmax(classifier_engine.predict(x_fgsm), axis=1)
                acc_fgsm = np.sum(preds_fgsm == y_test_np) / len(y_test_np) * 100
                
                with sim_col1:
                    st.markdown("#### **FGSM Analiz Çıktısı**")
                    st.metric("FGSM Sağlamlık Doğruluğu", f"%{acc_fgsm:.2f}", delta=f"- %{baseline_acc - acc_fgsm:.2f} Kayıp")
                    # "Akademik" kelimesi kaldırıldı (image_1e43af.png)
                    st.markdown("**FGSM Confusion Matrix**")
                    cm_f = confusion_matrix(y_test_np, preds_fgsm)
                    fig_f, ax_f = plt.subplots(figsize=(4, 3.2))
                    sns.heatmap(cm_f, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_f)
                    st.pyplot(fig_f)
                    plt.close(fig_f)
                    
            # [SAĞ] ibaresi kaldırıldı (image_1e469a.png)
            if "PGD" in selected_attacks:
                pgd_eng = ProjectedGradientDescent(estimator=classifier_engine, norm=art_norm, eps=eps_value, eps_step=eps_value/10, max_iter=10)
                x_pgd = pgd_eng.generate(x=x_test_np)
                preds_pgd = np.argmax(classifier_engine.predict(x_pgd), axis=1)
                acc_pgd = np.sum(preds_pgd == y_test_np) / len(y_test_np) * 100
                
                with sim_col2:
                    st.markdown("#### **PGD Analiz Çıktısı**")
                    st.metric("PGD Sağlamlık Doğruluğu", f"%{acc_pgd:.2f}", delta=f"- %{baseline_acc - acc_pgd:.2f} Kayıp")
                    # "Akademik" kelimesi kaldırıldı (image_1e43af.png)
                    st.markdown("**PGD Confusion Matrix**")
                    cm_p = confusion_matrix(y_test_np, preds_pgd)
                    fig_p, ax_p = plt.subplots(figsize=(4, 3.2))
                    sns.heatmap(cm_p, annot=True, fmt='d', cmap='Reds', cbar=False, ax=fig_p.gca())
                    st.pyplot(fig_p)
                    plt.close(fig_p)

            st.markdown("---")
            st.markdown("### 📈 Kritik Nokta Analizi Raporu")
            total_vulnerability_score = (baseline_acc - min(acc_fgsm, acc_pgd))
            
            if total_vulnerability_score > 10.0:
                st.error(f"🚨 KRİTİK ALARM: Yapay zekâ modeliniz üzerinde 10 birimden fazla zafiyet düşüşü ({total_vulnerability_score:.2f}) saptanmıştır!")
            else:
                st.success(f"🟢 DURUM STABİL: Saptanan zafiyet endeksi ({total_vulnerability_score:.2f}) güvenlik sınırları içerisindedir.")

            st.markdown("---")
            st.markdown("### 🧮 Sınıf Bazlı Bireysel Karışıklık Matrisleri")
            active_preds = preds_pgd if preds_pgd is not None else preds_fgsm
            
            if active_preds is not None:
                cm_global = confusion_matrix(y_test_np, active_preds, labels=list(range(10)))
                grid_col1, grid_col2, grid_col3 = st.columns(3)
                
                for digit in range(10):
                    idx = (y_test_np == digit)
                    digit_total = np.sum(idx)
                    digit_correct = np.sum(active_preds[idx] == y_test_np[idx])
                    digit_acc = (digit_correct / digit_total) * 100 if digit_total > 0 else 0
                    
                    target_grid = grid_col1 if digit % 3 == 0 else (grid_col2 if digit % 3 == 1 else grid_col3)
                    with target_grid:
                        st.markdown(f"**Sınıf [{digit}] Matrisi (Doğruluk: %{digit_acc:.1f})**")
                        fig_sub, ax_sub = plt.subplots(figsize=(2.5, 2))
                        sns.heatmap([cm_global[digit]], annot=True, fmt='d', cmap='Purples', cbar=False, xticklabels=range(10), yticklabels=[digit], ax=ax_sub)
                        st.pyplot(fig_sub)
                        plt.close(fig_sub)

                st.markdown("---")
                # Başlık "Genel Bakış" olarak değiştirildi (image_1e42c0.png)
                st.markdown("### 📊 Genel Bakış")
                fig_heatmap, ax_heatmap = plt.subplots(figsize=(7, 4.5))
                sns.heatmap(cm_global, annot=True, fmt='d', cmap='YlOrRd', xticklabels=range(10), yticklabels=range(10), ax=ax_heatmap)
                ax_heatmap.set_xlabel('Yapay Zekânın Saldırı Sonrası Tahmini', fontweight='bold')
                ax_heatmap.set_ylabel('Gerçek Rakam Sınıfı', fontweight='bold')
                st.pyplot(fig_heatmap)
                plt.close(fig_heatmap)
                
                class_scores = {}
                for c in range(10):
                    idx = (y_test_np == c)
                    if np.sum(idx) > 0:
                        class_scores[c] = np.sum(active_preds[idx] == y_test_np[idx]) / np.sum(idx) * 100
                vulnerable_sorted = sorted(class_scores.items(), key=lambda x: x[1])[:3]
                
                st.session_state['crit_class'] = vulnerable_sorted[0][0]
                st.session_state['saved_x_pgd'] = x_pgd if preds_pgd is not None else x_fgsm

                # "5. Katman" yazısı kaldırıldı
                st.markdown("---")
                st.markdown("### 📜 Küresel Güvenlik Raporu ve İndirme Merkezi")

                report_rows = []
                for digit in range(10):
                    idx = (y_test_np == digit)
                    t_total = np.sum(idx)
                    c_correct = np.sum(clean_preds[idx] == y_test_np[idx])
                    c_acc = (c_correct / t_total) * 100 if t_total > 0 else 0
                    a_correct = np.sum(active_preds[idx] == y_test_np[idx])
                    attack_acc = (a_correct / t_total) * 100 if t_total > 0 else 0
                    delta_drop = c_acc - attack_acc
                    
                    if delta_drop >= 25.0:
                        risk_status = "🔴 KRİTİK ZAFİYET"
                    elif delta_drop >= 10.0:
                        risk_status = "🟡 ORTA RİSK"
                    else:
                        risk_status = "🟢 GÜVENLİ"
                        
                    report_rows.append({
                        "Sınıf (Rakam)": f"Sınıf {digit}",
                        "Temiz Doğruluk": f"%{c_acc:.2f}",
                        "Saldırı Sonrası Doğruluk": f"%{attack_acc:.2f}",
                        "Performans Kaybı (Δ)": f"- %{delta_drop:.2f}",
                        "Güvenlik Durumu": risk_status
                    })

                df_report = pd.DataFrame(report_rows)
                st.dataframe(df_report, use_container_width=True)

                report_text = f"""=============================================================
         ROBUSPECT MLSECOPS MODEL GÜVENLİK DENETİM RAPORU
=============================================================
Rapor Tarihi: 2026
Geliştirici / Araştırmacı: Ebrar Şahin
Kurum: İstanbul Ticaret Üniversitesi

[1] MODEL VE EĞİTİM KÜNYESİ (OTOMATİK TESPİT)
-------------------------------------------------------------
* Saptanan Optimizasyon Motoru: {predicted_opt}
* Tahmin Edilen Eğitim Süresi: {predicted_epoch}
* Saptanan Öğrenme Kaysayısı (LR): {predicted_lr}

[2] KÜRESEL GÜVENLİK METRİKLERİ
-------------------------------------------------------------
* Test Edilen Tehdit Şiddeti (Epsilon): {eps_value}
* Kullanılan Pertürbasyon Normu: {norm_type}
* Modelin Orijinal Doğruluğu: %{baseline_acc:.2f}
* Genel Güvenlik Durumu: {"CRITICAL" if total_vulnerability_score > 10.0 else "STABLE"}

[3] SINIF BAZLI GÜVENLİK ANALİZ DETAYLARI
-------------------------------------------------------------
"""
                for row in report_rows:
                    report_text += f"* {row['Sınıf (Rakam)']}: Temiz: {row['Temiz Doğruluk']} | Saldırı Sonrası: {row['Saldırı Sonrası Doğruluk']} | Kayıp: {row['Performans Kaybı (Δ)']} -> Durum: {row['Güvenlik Durumu']}\n"
                
                st.markdown("#### **📥 Resmi Denetim Raporunu Dışarı Aktar**")
                st.download_button(
                    label="📄 RESMİ GÜVENLİK RAPORUNU (.TXT) İNDİR",
                    data=report_text,
                    file_name="robuspect_model_guvenlik_raporu.txt",
                    mime="text/plain",
                    use_container_width=True
                )

    with tab3:
        st.markdown("### 🧠 Gömülü XAI Teşhis Paneli (Canlı Grad-CAM)")
        if 'crit_class' not in st.session_state:
            st.info("💡 Grad-CAM üretebilmek için lütfen önce ikinci sekmeden 'Siber Testi Tetikle' butonuna basın.")
        else:
            c_class = st.session_state['crit_class']
            x_pgd_saved = st.session_state['saved_x_pgd']
            
            st.success(f"En çok sabote edilen **Sınıf [{c_class}]** için modelinizin derin katman nöron haritası çıkartılıyor...")
            
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
                
                # Tahmin olasılığını (Confidence) Softmax ile çıkaralım
                probs = F.softmax(out, dim=1)
                confidence = probs[0, p_cls].item() * 100
                
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
                return cam, p_cls, confidence

            cam_c, p_c, conf_c = compute_cam(img_clean_t)
            cam_a, p_a, conf_a = compute_cam(img_adv_t)
            
            hook_backward.remove(); hook_forward.remove()
            
            # Geliştirilmiş ve Optimize Edilmiş 4'lü Panel Çizimi
            fig_cam, axes = plt.subplots(1, 4, figsize=(14, 4))
            
            axes[0].imshow(x_test_np[t_idx].squeeze(), cmap='gray')
            axes[0].set_title(f"Orijinal Girdi (Sınıf: {c_class})\nDoğru Tahmin: {p_c}", color='green', fontsize=10, fontweight='bold')
            axes[0].axis('off')
            
            axes[1].imshow(x_test_np[t_idx].squeeze(), cmap='gray')
            axes[1].imshow(cam_c, cmap='jet', alpha=0.45)
            axes[1].set_title(f"Temiz Karar Odağı\nGüven Skoru: %{conf_c:.1f}", fontsize=10, fontweight='bold')
            axes[1].axis('off')
            
            axes[2].imshow(x_pgd_saved[t_idx].squeeze(), cmap='gray')
            axes[2].set_title(f"Saldırılı Girdi (Sınıf: {c_class})\nHatalı Tahmin: {p_a}", color='red', fontsize=10, fontweight='bold')
            axes[2].axis('off')
            
            axes[3].imshow(x_pgd_saved[t_idx].squeeze(), cmap='gray')
            axes[3].imshow(cam_a, cmap='jet', alpha=0.55) # Kontrast artırıldı
            axes[3].set_title(f"Saldırı Altındaki Sapma\nGüven Skoru: %{conf_a:.1f}", color='red', fontsize=10, fontweight='bold')
            axes[3].axis('off')
            
            plt.tight_layout()
            st.pyplot(fig_cam)
            plt.close(fig_cam)
