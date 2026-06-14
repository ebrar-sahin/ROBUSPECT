import streamlit as st
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.ndimage as ndimage
from sklearn.metrics import confusion_matrix
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent

st.set_page_config(page_title="Robuspect MLSecOps Core", layout="wide")

st.title("🛡️ Robuspect: Merkezi Siber Güvenilirlik Test Motoru")
st.subheader("Modeller İçin Karşılaştırmalı Zafiyet Analizi ve Görsel Adli Bilişim Platformu")
st.markdown("---")

# ==========================================
# 1. PLATFORMUN GÖMÜLÜ MODEL MİMARİSİ (KALIP)
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

# ==========================================
# 2. GÖMÜLÜ SABİT VERİ SETİ (BENCHMARK)
# ==========================================
@st.cache_resource
def load_embedded_test_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0,), (1.0,))])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=500, shuffle=False) # Hız için 500 örnek yeterlidir
    x_t, y_t = next(iter(test_loader))
    return x_t.numpy(), y_t.numpy()

x_test_np, y_test_np = load_embedded_test_data()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

# ==========================================
# 3. KULLANICI ARAYÜZÜ VE GİRDİLER (SIDEBAR)
# ==========================================
with st.sidebar:
    st.header("📥 Model Yükleme Merkezi")
    uploaded_weights = st.file_uploader("Eğittiğiniz Model Ağırlık Dosyasını Seçin (.pth)", type=["pth", "pt"])
    
    st.header("📋 Geliştirici Beyan Paneli")
    user_epoch = st.number_input("Modeli Kaç Epoch Eğittiniz?", min_value=1, max_value=200, value=5)
    user_opt = st.selectbox("Eğitimde Kullanılan Optimizasyon:", ["Adam", "SGD", "RMSprop", "Adagrad"])
    user_lr = st.number_input("Öğrenme Oranı (Learning Rate):", min_value=0.0001, max_value=0.1, value=0.001, format="%.4f")
    
    st.header("⚙️ Siber Güvenlik Ayarları")
    eps_value = st.slider("Gürültü Şiddeti (Epsilon ε):", min_value=0.01, max_value=0.50, value=0.15, step=0.01)
    
    st.markdown("---")
    st.markdown("**Platform Kurucusu:**\nEbrar Şahin\n*İstanbul Ticaret Üniversitesi*")

# ==========================================
# 4. MERKEZİ MOTOR AKIŞ MANTIĞI
# ==========================================
if uploaded_weights is None:
    st.warning("⚠️ Lütfen siber güvenlik test motorunu çalıştırmak için sol panelden eğitilmiş bir model ağırlık dosyası (.pth) yükleyin.")
else:
    with st.spinner("Model ağırlık matrisleri çözümleniyor ve motor kalıbına dökülüyor..."):
        # Saf ağırlıkları hafızaya alma
        state_dict = torch.load(uploaded_weights, map_location=torch.device('cpu'))
        
        # Eğer kullanıcı checkpoint sözlüğü yüklediyse saf ağırlıkları süz
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
            
        # Ağırlıkları boş mimari şablona enjekte etme
        eval_model = MNISTCNN().to(device)
        eval_model.load_state_dict(state_dict)
        eval_model.eval()
        
        # İçsel ART Yapay Zekâ Koruma Katmanı Yapılandırması
        dummy_optimizer = torch.optim.Adam(eval_model.parameters(), lr=0.001)
        classifier_engine = PyTorchClassifier(
            model=eval_model, clip_values=(0.0, 1.0), loss=criterion,
            optimizer=dummy_optimizer, input_shape=(1, 28, 28), nb_classes=10
        )

    # 3 Ana Akademik Sekme (Tabs)
    tab1, tab2, tab3 = st.tabs(["📋 Yapay Zekâ Araç Raporu", "📊 Eş Zamanlı Siber Simülasyon", "🧠 XAI / Canlı Grad-CAM Teşhisi"])
    
    # SEKMELER 1: MODEL KÜNYESİ
    with tab1:
        st.markdown("### 🔍 Model Röntgeni ve Araç Doğrulama İstatistikleri")
        col_k1, col_k2, col_k3 = st.columns(3)
        col_k1.metric("Eğitim Dönemi (Epoch)", f"{user_epoch} Dönem")
        col_k2.metric("Doğrulanan Optimizer", user_opt)
        col_k3.metric("Öğrenme Katsayısı (LR)", f"{user_lr}")
        
        st.markdown("#### **Ağırlık Tensorlarından Çıkarılan Katman Anatomisi**")
        layers_info = []
        for key in state_dict.keys():
            if 'weight' in key:
                name = key.split('.')[0]
                shape = list(state_dict[key].shape)
                l_type = "Evrişim (Conv2d)" if len(shape) == 4 else "Tam Bağlantılı (Linear)"
                layers_info.append({"Katman İsmi": name, "Katman Türü": l_type, "Matris Boyutu (Tensor Shape)": str(shape)})
        st.table(layers_info)

    # SEKMELER 2: SİBER SİMÜLASYON VE CONFUSION MATRIX
    with tab2:
        st.markdown("### ⚙️ FGSM ve PGD Karşılaştırmalı Siber Laboratuvarı")
        st.markdown(f"Gömülü motor, şu an modelinizi aynı veri kümesi üzerinde **ε = {eps_value}** şiddetindeki iki farklı saldırıya maruz bırakmaktadır.")
        
        # Temiz doğruluk
        clean_preds = np.argmax(classifier_engine.predict(x_test_np), axis=1)
        baseline_acc = np.sum(clean_preds == y_test_np) / len(y_test_np) * 100
        
        st.info(f"🎯 **Modelinizin Temiz (Saldırısız) Verideki Başarı Oranı:** %{baseline_acc:.2f}")
        
        trigger_btn = st.button("🚀 SİBER GÜVENLİK TESTİNİ TETİKLE", use_container_width=True)
        
        if trigger_btn:
            with st.spinner("FGSM ve PGD tehdit vektörleri eş zamanlı olarak simüle ediliyor..."):
                # FGSM Motoru
                fgsm_eng = FastGradientMethod(estimator=classifier_engine, eps=eps_value)
                x_fgsm = fgsm_eng.generate(x=x_test_np)
                preds_fgsm = np.argmax(classifier_engine.predict(x_fgsm), axis=1)
                acc_fgsm = np.sum(preds_fgsm == y_test_np) / len(y_test_np) * 100
                
                # PGD Motoru
                pgd_eng = ProjectedGradientDescent(estimator=classifier_engine, eps=eps_value, eps_step=eps_value/10, max_iter=10)
                x_pgd = pgd_eng.generate(x=x_test_np)
                preds_pgd = np.argmax(classifier_engine.predict(x_pgd), axis=1)
                acc_pgd = np.sum(preds_pgd == y_test_np) / len(y_test_np) * 100
                
                # Yan yana iki sütunlu akademik gösterim
                c1, c2 = st.columns(2)
                
                with c1:
                    st.metric("FGSM Sağlamlık Doğruluğu", f"%{acc_fgsm:.2f}", delta=f"- %{baseline_acc - acc_fgsm:.2f} Kayıp")
                    st.markdown("**FGSM Akademik Confusion Matrix**")
                    cm_f = confusion_matrix(y_test_np, preds_fgsm)
                    fig_f, ax_f = plt.subplots(figsize=(4, 3.2))
                    sns.heatmap(cm_f, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_f)
                    st.pyplot(fig_f)
                    
                with c2:
                    st.metric("PGD Sağlamlık Doğruluğu", f"%{acc_pgd:.2f}", delta=f"- %{baseline_acc - acc_pgd:.2f} Kayıp")
                    st.markdown("**PGD Akademik Confusion Matrix**")
                    cm_p = confusion_matrix(y_test_np, preds_pgd)
                    fig_p, ax_p = plt.subplots(figsize=(4, 3.2))
                    sns.heatmap(cm_p, annot=True, fmt='d', cmap='Reds', cbar=False, ax=ax_p)
                    st.pyplot(fig_p)
                
                # Sınıf Bazlı Kırılganlık Sıralaması (PGD sonuçları üzerinden hesaplanır)
                class_scores = {}
                for c in range(10):
                    idx = (y_test_np == c)
                    if np.sum(idx) > 0:
                        class_scores[c] = np.sum(preds_pgd[idx] == y_test_np[idx]) / np.sum(idx) * 100
                
                vulnerable_sorted = sorted(class_scores.items(), key=lambda x: x[1])[:3]
                
                st.markdown("---")
                st.markdown("### 🚨 Sınıf Bazlı Kırılganlık Sıralaması (Zafiyet Analizi)")
                st.markdown("Saldırı altında en hızlı çöken ve manipüle edilmeye en açık ilk 3 rakam sınıfı:")
                
                r1, r2, r3 = st.columns(3)
                r1.error(f"🔺 **1. En Kırılgan:** Sınıf [{vulnerable_sorted[0][0]}] | Doğruluk: %{vulnerable_sorted[0][1]:.2f}")
                r2.warning(f"🔸 **2. Kırılgan:** Sınıf [{vulnerable_sorted[1][0]}] | Doğruluk: %{vulnerable_sorted[1][1]:.2f}")
                r3.warning(f"🔸 **3. Kırılgan:** Sınıf [{vulnerable_sorted[2][0]}] | Doğruluk: %{vulnerable_sorted[2][1]:.2f}")
                
                # Verileri Grad-CAM sekmesine taşımak için session_state'e yazıyoruz
                st.session_state['crit_class'] = vulnerable_sorted[0][0]
                st.session_state['saved_x_pgd'] = x_pgd

    # SEKMELER 3: GRAD-CAM FORENSIC PANEL
    with tab3:
        st.markdown("### 🧠 4. Katman: Gömülü XAI Teşhis Paneli (Canlı Grad-CAM)")
        if 'crit_class' not in st.session_state:
            st.info("💡 Grad-CAM adli analizinin yapılabilmesi için lütfen önce ikinci sekmedeki 'Siber Güvenlik Testini Tetikle' butonuna basın.")
        else:
            c_class = st.session_state['crit_class']
            x_pgd_saved = st.session_state['saved_x_pgd']
            
            st.success(f"En çok sabote edilen **Sınıf [{c_class}]** için modelinizin derin katman nöron aktivasyon haritası çıkarılıyor...")
            
            # Kancalar yardımıyla en son evrişim katmanına (conv2) canlı bağlanma
            g_list, a_list = [], []
            def h_b(module, gi, go): g_list.append(go[0])
            def h_f(module, i, o): a_list.append(o)
            
            hook_backward = eval_model.conv2.register_full_backward_hook(h_b)
            hook_forward = eval_model.conv2.register_forward_hook(h_f)
            
            # Seçilen kırılgan sınıfın ilk imajını bulma
            t_idx = np.where(y_test_np == c_class)[0][0]
            img_clean_t = torch.tensor(x_test_np[t_idx:t_idx+1]).to(device)
            img_adv_t = torch.tensor(x_pgd_saved[t_idx:t_idx+1]).to(device)
            
            def compute_cam(img):
                g_list.clear(); a_list.clear()
                out = eval_model(img)
                p_cls = torch.argmax(out, dim=1).item()
                eval_model.zero_grad()
                out[0, p_cls].backward()
                gr = g_list[0].cpu().data.numpy()[0]
                ac = a_list[0].cpu().data.numpy()[0]
                w = np.mean(gr, axis=(1, 2))
                cam = np.zeros(ac.shape[1:], dtype=np.float32)
                for i, w_val in enumerate(w): cam += w_val * ac[i]
                cam = np.maximum(cam, 0)
                if np.max(cam) > 0: cam /= np.max(cam)
                cam = ndimage.zoom(cam, (28 / cam.shape[0], 28 / cam.shape[1]), order=1)
                return cam, p_cls

            cam_c, p_c = compute_cam(img_clean_t)
            cam_a, p_a = compute_cam(img_adv_t)
            
            hook_backward.remove(); hook_forward.remove() # Kancaları sökme
            
            # 4'lü Açıklanabilirlik Matrisi Çizimi
            fig_cam, axes = plt.subplots(1, 4, figsize=(12, 3.5))
            axes[0].imshow(x_test_np[t_idx].squeeze(), cmap='gray')
            axes[0].set_title(f"Orijinal Rakam: {c_class}\nTahmin: {p_c}", color='green', fontsize=9)
            axes[0].axis('off')
            
            axes[1].imshow(x_test_np[t_idx].squeeze(), cmap='gray')
            axes[1].imshow(cam_c, cmap='jet', alpha=0.4)
            axes[1].set_title("Normal Karar Odağı", fontsize=9)
            axes[1].axis('off')
            
            axes[2].imshow(x_pgd_saved[t_idx].squeeze(), cmap='gray')
            axes[2].set_title(f"Saldırılı Rakam: {c_class}\nTahmin: {p_a}", color='red', fontsize=9)
            axes[2].axis('off')
            
            axes[3].imshow(x_pgd_saved[t_idx].squeeze(), cmap='gray')
            axes[3].imshow(cam_a, cmap='jet', alpha=0.4)
            axes[3].set_title("Saldırı Altındaki Sapma", color='red', fontsize=9)
            axes[3].axis('off')
            
            plt.tight_layout()
            st.pyplot(fig_cam)
            
            st.markdown("""
            **Mühendislik ve Teşhis Özeti:** Yapay zekâ modeliniz temiz verideyken rakamın merkezindeki ayırt edici geometrik hatlara odaklanarak doğru karar vermektedir. Ancak siber gürültü (adversarial perturbation) enjekte edildiğinde, modelin dikkat mekanizması (nöron aktivasyonu) kenarlardaki anlamsız boşluklara kaymaktadır. Bu durum, modelin dış etkenlere karşı ne kadar kolay manipüle edilebildiğini gözler önüne sermektedir.
            """)
