from PyQt5.QtWidgets import *
from PyQt5.QtGui import QFont, QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from ultralytics import YOLO
from database import simpan_deteksi
import cv2


# =========================
# THREAD UNTUK DETEKSI YOLO
# =========================
class DetectWorker(QThread):
    result_ready = pyqtSignal(object, object, object)

    def __init__(self, model, image_path):
        super().__init__()
        self.model = model
        self.image_path = image_path

    def run(self):
        try:
            results = self.model.predict(
                source=self.image_path,
                conf=0.25,
                iou=0.5
            )

            img = results[0].plot()
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()

            detections = []

            if len(results[0].boxes) > 0:
                for i, box in enumerate(results[0].boxes, start=1):
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = self.model.names[cls_id]

                    detections.append({
                        "no": i,
                        "class_name": class_name,
                        "confidence": conf
                    })

                self.result_ready.emit(img_rgb, detections, None)

            else:
                self.result_ready.emit(img_rgb, [], None)

        except Exception as e:
            self.result_ready.emit(None, [], str(e))


# =========================
# THREAD UNTUK SIMPAN DATABASE
# =========================
class SaveDBWorker(QThread):
    save_finished = pyqtSignal(bool, str)

    def __init__(self, detections):
        super().__init__()
        self.detections = detections

    def run(self):
        try:
            total_data = len(self.detections)
            total_berhasil = 0
            pesan_gagal = []

            for data in self.detections:
                berhasil, pesan = simpan_deteksi(
                    data["class_name"],
                    data["confidence"]
                )

                if berhasil:
                    total_berhasil += 1
                else:
                    pesan_gagal.append(f"Objek {data['no']}: {pesan}")

            if total_berhasil == total_data:
                self.save_finished.emit(
                    True,
                    f"{total_berhasil} hasil deteksi berhasil disimpan ke database"
                )
            else:
                self.save_finished.emit(
                    False,
                    f"{total_berhasil} dari {total_data} hasil deteksi berhasil disimpan. "
                    + " ".join(pesan_gagal)
                )

        except Exception as e:
            self.save_finished.emit(False, str(e))


# =========================
# MAIN WINDOW
# =========================
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Deteksi Kekeringan Rumput Laut")
        self.setGeometry(100, 100, 1000, 600)

        self.model = YOLO(r"D:\KULIAH\SEMESTER_8\SKRIPSIP_ZIDAN\all\runs\detect\yolov8n\weights\best.pt")
        self.image_path = None
        self.detect_worker = None
        self.db_worker = None
        self._force_close = False

        # ===== STYLE APLIKASI =====
        self.apply_style()

        # ===== HEADER =====
        self.title = QLabel("Sistem Klasifikasi Kekeringan Rumput Laut")
        self.title.setObjectName("titleLabel")
        self.title.setFont(QFont("Arial", 18, QFont.Bold))
        self.title.setAlignment(Qt.AlignCenter)

        # ===== CONTROL PANEL =====
        control_group = QGroupBox("Kontrol")
        control_layout = QVBoxLayout()

        self.btn_upload = QPushButton("📂 Upload Gambar")
        self.btn_detect = QPushButton("🔍 Deteksi")
        self.btn_reset = QPushButton("♻ Reset")
        self.btn_about = QPushButton("ℹ Tentang")
        self.btn_guide = QPushButton("📖 Panduan")
        self.btn_exit = QPushButton("🚪 Keluar")

        self.btn_upload.setObjectName("primaryButton")
        self.btn_detect.setObjectName("successButton")
        self.btn_exit.setObjectName("dangerButton")

        for btn in [
            self.btn_upload,
            self.btn_detect,
            self.btn_reset,
            self.btn_about,
            self.btn_guide,
            self.btn_exit
        ]:
            btn.setFixedHeight(40)

        control_layout.addWidget(self.btn_upload)
        control_layout.addWidget(self.btn_detect)
        control_layout.addWidget(self.btn_reset)
        control_layout.addWidget(self.btn_about)
        control_layout.addWidget(self.btn_guide)
        control_layout.addStretch()
        control_layout.addWidget(self.btn_exit)

        control_group.setLayout(control_layout)

        # ===== IMAGE PREVIEW =====
        self.image_label = QLabel("📷\nBelum ada gambar")
        self.image_label.setObjectName("imagePreview")
        self.image_label.setAlignment(Qt.AlignCenter)

        # ===== RESULT PANEL =====
        result_group = QGroupBox("Hasil Deteksi")
        result_layout = QVBoxLayout()

        self.label_class = QLabel("Kelas: -")
        self.label_conf = QLabel("Akurasi: -")
        self.label_detail = QLabel("Detail Deteksi: -")
        self.label_status = QLabel("Status: Menunggu gambar")

        self.label_detail.setWordWrap(True)
        self.label_status.setWordWrap(True)

        self.label_class.setObjectName("classLabel")
        self.label_conf.setObjectName("confLabel")
        self.label_detail.setObjectName("statusLabel")
        self.label_status.setObjectName("statusLabel")

        result_layout.addWidget(self.label_class)
        result_layout.addWidget(self.label_conf)
        result_layout.addWidget(self.label_detail)
        result_layout.addWidget(self.label_status)
        result_layout.addStretch()

        result_group.setLayout(result_layout)

        # ===== MAIN CONTENT =====
        content_layout = QHBoxLayout()
        content_layout.setSpacing(15)

        content_layout.addWidget(control_group, 1)
        content_layout.addWidget(self.image_label, 3)
        content_layout.addWidget(result_group, 1)

        # ===== MAIN LAYOUT =====
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.title)
        main_layout.addLayout(content_layout)

        self.setLayout(main_layout)

        # ===== CONNECT BUTTON =====
        self.btn_upload.clicked.connect(self.upload_image)
        self.btn_detect.clicked.connect(self.detect_image)
        self.btn_reset.clicked.connect(self.reset_all)
        self.btn_about.clicked.connect(self.show_about)
        self.btn_guide.clicked.connect(self.show_guide)
        self.btn_exit.clicked.connect(self.exit_app)

    # =========================
    # STYLE APLIKASI
    # =========================
    def apply_style(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #F5F7F8;
                color: #263238;
                font-family: Arial;
                font-size: 10.5pt;
            }

            QLabel#titleLabel {
                color: #1F3A3D;
                padding: 14px 0px;
                border-bottom: 2px solid #B7C9C4;
            }

            QGroupBox {
                background-color: #FFFFFF;
                border: 1px solid #D6DEDC;
                border-radius: 10px;
                margin-top: 12px;
                padding: 14px 10px 10px 10px;
                font-weight: bold;
                color: #1F3A3D;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0px 6px;
                background-color: #F5F7F8;
            }

            QPushButton {
                background-color: #ECEFF1;
                color: #263238;
                border: 1px solid #CFD8DC;
                border-radius: 8px;
                padding: 8px 10px;
                text-align: left;
                font-weight: 600;
            }

            QPushButton:hover {
                background-color: #DDE6E4;
                border: 1px solid #AEBFBB;
            }

            QPushButton:disabled {
                background-color: #E0E0E0;
                color: #9E9E9E;
                border: 1px solid #D0D0D0;
            }

            QPushButton#primaryButton {
                background-color: #2F6F73;
                color: white;
                border: 1px solid #2F6F73;
            }

            QPushButton#primaryButton:hover {
                background-color: #285E61;
            }

            QPushButton#successButton {
                background-color: #4F8A5B;
                color: white;
                border: 1px solid #4F8A5B;
            }

            QPushButton#successButton:hover {
                background-color: #42764C;
            }

            QPushButton#dangerButton {
                background-color: #B55252;
                color: white;
                border: 1px solid #B55252;
            }

            QPushButton#dangerButton:hover {
                background-color: #994646;
            }

            QLabel#imagePreview {
                background-color: #FFFFFF;
                border: 2px dashed #B7C9C4;
                border-radius: 12px;
                color: #78909C;
                font-size: 13pt;
            }

            QLabel#classLabel {
                font-weight: bold;
                font-size: 15px;
                color: #1F3A3D;
            }

            QLabel#confLabel {
                color: #4F8A5B;
                font-weight: bold;
            }

            QLabel#statusLabel {
                color: #546E7A;
                line-height: 140%;
            }
        """)

    # =========================
    # UPLOAD GAMBAR
    # =========================
    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Pilih Gambar",
            "",
            "Image Files (*.png *.jpg *.jpeg)"
        )

        if file_path:
            self.image_path = file_path

            pixmap = QPixmap(file_path)
            self.image_label.setPixmap(
                pixmap.scaled(
                    self.image_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
            )

            self.label_class.setText("Kelas: -")
            self.label_conf.setText("Akurasi: -")
            self.label_detail.setText("Detail Deteksi: -")
            self.label_status.setText("Status: Gambar berhasil diupload")

    # =========================
    # DETEKSI GAMBAR
    # =========================
    def detect_image(self):
        if not self.image_path:
            self.label_status.setText("Status: Silahkan Upload gambar terlebih dahulu")
            return

        self.btn_detect.setEnabled(False)
        self.label_status.setText("Status: Proses deteksi...")

        self.detect_worker = DetectWorker(self.model, self.image_path)
        self.detect_worker.result_ready.connect(self.on_detection_finished)
        self.detect_worker.start()

    # =========================
    # SETELAH DETEKSI SELESAI
    # =========================
    def on_detection_finished(self, img_rgb, detections, error):
        self.btn_detect.setEnabled(True)

        if error is not None:
            QMessageBox.critical(self, "Error Deteksi", error)
            self.label_status.setText("Status: Terjadi error saat deteksi")
            return

        if img_rgb is not None:
            h, w, ch = img_rgb.shape
            bytes_per_line = ch * w

            qt_image = QImage(
                img_rgb.data,
                w,
                h,
                bytes_per_line,
                QImage.Format_RGB888
            ).copy()

            self.image_label.setPixmap(
                QPixmap.fromImage(qt_image).scaled(
                    self.image_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
            )

        if len(detections) > 0:
            kelas_unik = []

            for data in detections:
                if data["class_name"] not in kelas_unik:
                    kelas_unik.append(data["class_name"])

            akurasi_tertinggi = max(data["confidence"] for data in detections)

            detail_text = f"Jumlah objek: {len(detections)}\n"

            for data in detections:
                detail_text += (
                    f"Objek {data['no']}: "
                    f"{data['class_name']} "
                    f"({data['confidence']:.2f})\n"
                )

            self.label_class.setText(f"Kelas: {', '.join(kelas_unik)}")
            self.label_conf.setText(f"Akurasi tertinggi: {akurasi_tertinggi:.2f}")
            self.label_detail.setText(detail_text)
            self.label_status.setText("Status: Deteksi berhasil, menyimpan ke database...")

            self.simpan_database(detections)

        else:
            self.label_class.setText("Kelas: Tidak terdeteksi")
            self.label_conf.setText("Akurasi: -")
            self.label_detail.setText("Jumlah objek: 0")
            self.label_status.setText("Status: Tidak ada objek terdeteksi")

    # =========================
    # SIMPAN DATABASE
    # =========================
    def simpan_database(self, detections):
        self.db_worker = SaveDBWorker(detections)
        self.db_worker.save_finished.connect(self.on_save_finished)
        self.db_worker.start()

    # =========================
    # SETELAH SIMPAN DATABASE
    # =========================
    def on_save_finished(self, berhasil, pesan):
        print(pesan)

        if berhasil:
            self.label_status.setText("Status: Semua hasil deteksi berhasil & tersimpan ke database")
        else:
            self.label_status.setText("Status: Deteksi berhasil, tetapi sebagian/gagal simpan database")
            QMessageBox.warning(self, "Database", pesan)

    # =========================
    # RESET
    # =========================
    def reset_all(self):
        self.image_label.clear()
        self.image_label.setText("📷\nBelum ada gambar")
        self.image_label.setAlignment(Qt.AlignCenter)

        self.label_class.setText("Kelas: -")
        self.label_conf.setText("Akurasi: -")
        self.label_detail.setText("Detail Deteksi: -")
        self.label_status.setText("Status: Menunggu gambar")

        self.image_path = None

    # =========================
    # KELUAR APLIKASI
    # =========================
    def exit_app(self):
        if self.confirm_exit():
            self._force_close = True
            self.close()

    def confirm_exit(self):
        if self.detect_worker is not None and self.detect_worker.isRunning():
            QMessageBox.warning(
                self,
                "Proses Masih Berjalan",
                "Deteksi masih berjalan. Tunggu sampai proses selesai sebelum keluar."
            )
            return False

        pilihan = QMessageBox.question(
            self,
            "Konfirmasi Keluar",
            "Apakah Anda yakin ingin keluar dari aplikasi?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        return pilihan == QMessageBox.Yes

    def closeEvent(self, event):
        if self._force_close:
            event.accept()
            return

        if self.confirm_exit():
            event.accept()
        else:
            event.ignore()

    # =========================
    # POPUP TENTANG
    # =========================
    def show_about(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Tentang Aplikasi")
        dialog.setFixedSize(600, 420)

        main_layout = QVBoxLayout()

        # =========================
        # DATA DOSEN
        # =========================
        dosen_layout = QHBoxLayout()

        foto_dosen = QLabel()
        pixmap_dsn = QPixmap("ZIDANALMA2.jpg")

        if not pixmap_dsn.isNull():
            foto_dosen.setPixmap(
                pixmap_dsn.scaled(140, 170, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        else:
            foto_dosen.setText("Foto dosen\n tidak ditemukan")
            foto_dosen.setAlignment(Qt.AlignCenter)

        foto_dosen.setFixedSize(150, 180)
        foto_dosen.setAlignment(Qt.AlignCenter)
        foto_dosen.setStyleSheet("border: 1px solid #B7C9C4; border-radius: 6px; background-color: white;")

        biodata_dosen = QLabel(
            "Dosen Pembimbing\n\n"
            "Nama : Arif Fadllullah, S.Pd., M.Kom.\n"
            "NIP  : 199105192019031012"
        )
        biodata_dosen.setStyleSheet("font-size: 12pt; color: #263238;")
        biodata_dosen.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        dosen_layout.addWidget(foto_dosen)
        dosen_layout.addSpacing(15)
        dosen_layout.addWidget(biodata_dosen)

        # =========================
        # GARIS PEMBATAS
        # =========================
        garis = QFrame()
        garis.setFrameShape(QFrame.HLine)
        garis.setFrameShadow(QFrame.Sunken)

        # =========================
        # DATA MAHASISWA
        # =========================
        mhs_layout = QHBoxLayout()

        foto_mhs = QLabel()
        pixmap_mhs = QPixmap("ZIDANALMA2.jpg")

        if not pixmap_mhs.isNull():
            foto_mhs.setPixmap(
                pixmap_mhs.scaled(140, 170, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        else:
            foto_mhs.setText("Foto mahasiswa\n tidak ditemukan")
            foto_mhs.setAlignment(Qt.AlignCenter)

        foto_mhs.setFixedSize(150, 180)
        foto_mhs.setAlignment(Qt.AlignCenter)
        foto_mhs.setStyleSheet("border: 1px solid #B7C9C4; border-radius: 6px; background-color: white;")

        biodata_mhs = QLabel(
            "Mahasiswa\n\n"
            "Nama : Muhammad Zidan Fariz\n"
            "NPM  : 2240304032"
        )
        biodata_mhs.setStyleSheet("font-size: 12pt; color: #263238;")
        biodata_mhs.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        mhs_layout.addWidget(foto_mhs)
        mhs_layout.addSpacing(15)
        mhs_layout.addWidget(biodata_mhs)

        # =========================
        # SUSUN KE MAIN LAYOUT
        # =========================
        main_layout.addLayout(dosen_layout)
        main_layout.addWidget(garis)
        main_layout.addLayout(mhs_layout)

        dialog.setLayout(main_layout)
        dialog.exec_()

    # =========================
    # POPUP PANDUAN
    # =========================
    def show_guide(self):
        QMessageBox.information(
            self,
            "Panduan Penggunaan",
            "Cara menggunakan aplikasi:\n\n"
            "1. Klik 'Upload Gambar'\n"
            "2. Pilih gambar rumput laut\n"
            "3. Klik 'Deteksi'\n"
            "4. Lihat hasil deteksi pada panel kanan\n\n"
            "Catatan:\n"
            "Hasil deteksi disimpan ke database MySQL db_deteksi.\n"
            "Riwayat hasil deteksi dapat dilihat melalui phpMyAdmin.\n\n"
            "Tips:\n"
            "Gunakan gambar yang jelas agar hasil deteksi lebih akurat."
        )


if __name__ == "__main__":
    app = QApplication([])

    window = MainWindow()
    window.show()

    app.exec_()