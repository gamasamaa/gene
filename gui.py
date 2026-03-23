import sys
import os
import cv2 as cv
import numpy as np
import csv
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QFileDialog, 
                             QFormLayout, QSpinBox, QMessageBox, QScrollArea)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont
from gene import detect_dots_adaptive, map_dots_to_frame

class ImageDropLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText("\n\nDrag and Drop Image Here\n\nor Click 'Browse'\n\n")
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaa;
                border-radius: 8px;
                background-color: #f7f7f7;
                color: #555;
            }
        """)
        font = QFont()
        font.setPointSize(16)
        self.setFont(font)
        self.setAcceptDrops(True)
        self.setMinimumSize(400, 300)
        self.main_window = None

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls and urls[0].isLocalFile():
            file_path = urls[0].toLocalFile()
            if self.main_window:
                self.main_window.load_image(file_path)

class DotDetectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dot Detector GUI")
        self.resize(800, 600)
        self.current_image_path = None
        self.frame_path = None
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)

        # Left panel (Settings)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(250)

        # Parameters
        form_layout = QFormLayout()

        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(1, 100000)
        self.min_area_spin.setValue(500)
        
        self.max_area_spin = QSpinBox()
        self.max_area_spin.setRange(1, 500000)
        self.max_area_spin.setValue(12000)
        
        self.block_size_spin = QSpinBox()
        self.block_size_spin.setRange(3, 999)
        self.block_size_spin.setSingleStep(2)
        self.block_size_spin.setValue(151)
        
        self.c_spin = QSpinBox()
        self.c_spin.setRange(1, 100)
        self.c_spin.setValue(10)

        self.blur_size_spin = QSpinBox()
        self.blur_size_spin.setRange(3, 99)
        self.blur_size_spin.setSingleStep(2)
        self.blur_size_spin.setValue(5)

        form_layout.addRow("Min Area:", self.min_area_spin)
        form_layout.addRow("Max Area:", self.max_area_spin)
        form_layout.addRow("Block Size (odd):", self.block_size_spin)
        form_layout.addRow("C Value:", self.c_spin)
        form_layout.addRow("Blur Size (odd):", self.blur_size_spin)

        left_layout.addLayout(form_layout)

        # Buttons
        self.browse_btn = QPushButton("Browse Image")
        self.browse_btn.clicked.connect(self.browse_image)
        left_layout.addWidget(self.browse_btn)

        self.map_btn = QPushButton("Select Map File (Opt)")
        self.map_btn.clicked.connect(self.browse_map)
        left_layout.addWidget(self.map_btn)
        
        self.map_label = QLabel("Map: None")
        self.map_label.setWordWrap(True)
        left_layout.addWidget(self.map_label)

        self.process_btn = QPushButton("Process Image")
        self.process_btn.clicked.connect(self.process_image)
        self.process_btn.setEnabled(False)
        self.process_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        left_layout.addWidget(self.process_btn)

        self.status_label = QLabel("Status: Ready")
        left_layout.addWidget(self.status_label)
        
        left_layout.addStretch()

        # Right panel (Image display)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.image_label = ImageDropLabel()
        self.image_label.main_window = self
        
        # Scroll area for the image to handle large images gracefully
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.image_label)
        
        right_layout.addWidget(scroll_area)

        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)

    def browse_map(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Map Frame", "", "Text Files (*.txt);;All Files (*)"
        )
        if file_path:
            self.frame_path = file_path
            self.map_label.setText(f"Map: {os.path.basename(file_path)}")

    def browse_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.tiff *.tif *.bmp)"
        )
        if file_path:
            self.load_image(file_path)

    def load_image(self, file_path):
        self.current_image_path = file_path
        pixmap = QPixmap(file_path)
        if pixmap.isNull():
            QMessageBox.critical(self, "Error", "Could not load image.")
            return

        # Setup scaled down display if it's too huge, or we can just show it
        # Actually scaling it allows us to see the whole thing, but we also want to display circles properly.
        # So we'll update the label pixmap.
        self.display_image(pixmap, "Original Image Loaded")
        
        self.process_btn.setEnabled(True)
        self.status_label.setText(f"File loaded: {os.path.basename(file_path)}")

    def display_image(self, pixmap, status_text=""):
        # Scale the pixmap to fit the label size while keeping aspect ratio
        # For simplicity, we just set it. The scroll area Handles it.
        # It's better to scale it a bit if it's too big, e.g. max 1200x800
        if pixmap.width() > 1200 or pixmap.height() > 800:
            pixmap = pixmap.scaled(1200, 800, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

        self.image_label.setPixmap(pixmap)
        if status_text:
            self.status_label.setText(f"Status: {status_text}")

    def process_image(self):
        if not self.current_image_path:
            return

        # Validate block size is odd
        block_size = self.block_size_spin.value()
        if block_size % 2 == 0:
            block_size += 1
            self.block_size_spin.setValue(block_size)
            QMessageBox.warning(self, "Adjusted", "Block size must be odd. Adjusted to " + str(block_size))

        blur_size = self.blur_size_spin.value()
        if blur_size % 2 == 0:
            blur_size += 1
            self.blur_size_spin.setValue(blur_size)
            QMessageBox.warning(self, "Adjusted", "Blur size must be odd. Adjusted to " + str(blur_size))

        min_area = self.min_area_spin.value()
        max_area = self.max_area_spin.value()
        c_val = self.c_spin.value()

        # Prompt for save location
        base_name = os.path.splitext(self.current_image_path)[0]
        default_csv_path = f"{base_name}.csv"
        
        output_csv, _ = QFileDialog.getSaveFileName(
            self, "Save CSV As", default_csv_path, "CSV Files (*.csv)"
        )
        
        if not output_csv:
            self.status_label.setText("Status: Ready")
            return

        self.status_label.setText("Status: Processing...")
        QApplication.processEvents() # Force UI update

        try:
            # Call the gene.py function
            coords = detect_dots_adaptive(
                self.current_image_path, 
                min_area=min_area, 
                max_area=max_area, 
                block_size=block_size, 
                C=c_val,
                blur_size=blur_size
            )
            
            if self.frame_path:
                mapped_coords = map_dots_to_frame(coords, self.frame_path)
                with open(output_csv, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile, lineterminator='\n')
                    writer.writerow(['ID', 'Name', 'X Coordinate', 'Y Coordinate', 'Radius'])
                    for (x, y, r, p_id, p_name) in mapped_coords:
                        writer.writerow([p_id, p_name, x, y, r])
            else:
                with open(output_csv, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile, lineterminator='\n')
                    writer.writerow(['Dot ID', 'X Coordinate', 'Y Coordinate', 'Radius'])
                    for idx, (x, y, r) in enumerate(coords):
                        writer.writerow([idx + 1, x, y, r])
            
            # Draw on original image
            original_img = cv.imread(self.current_image_path)
            if original_img is not None:
                for idx, (x, y, r) in enumerate(coords):
                    cv.circle(original_img, (x, y), r, (0, 255, 0), 2)
                    cv.circle(original_img, (x, y), 2, (0, 0, 255), 3)
                
                # Convert BGR to RGB
                original_img = cv.cvtColor(original_img, cv.COLOR_BGR2RGB)
                h, w, ch = original_img.shape
                bytes_per_line = ch * w
                qt_img = QImage(original_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                
                pixmap = QPixmap.fromImage(qt_img)
                self.display_image(pixmap)

            self.status_label.setText(f"Status: Found {len(coords)} dots! Saved to {output_csv}")
            
            QMessageBox.information(self, "Success", f"Detected {len(coords)} dots.\nCoordinates saved to:\n{output_csv}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred:\n{str(e)}")
            self.status_label.setText("Status: Error")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Modern styling
    app.setStyle("Fusion")
    
    window = DotDetectorApp()
    window.show()
    sys.exit(app.exec())
