from __future__ import division

__author__ = 'tomas'

import sys
from PyQt4 import QtGui, QtCore

import numpy as np
import scipy.stats as scista
#import tools
import os
if os.path.exists('../imtools/'):
    sys.path.append('../imtools/')
    from imtools import tools
else:
    print 'You need to import package imtools: https://github.com/mjirik/imtools'
    sys.exit(0)

# from mayavi import mlab
# import TumorVisualiser
# import Viewer_3D
from os import path

import io3d
import ConfigParser
import cPickle as pickle
import gzip

import logging
logger = logging.getLogger(__name__)
logging.basicConfig()

# from lession_editor_GUI_slim import Ui_MainWindow
from seg_viewer_GUI import SegViewerGUI
# from hist_widget import Hist_widget
# from objects_widget import Objects_widget
# import My_table_model as mtm
# import area_hist_widget as ahw
# import computational_core as coco
import data_view_widget
import Lesion
import Data

# constants definition
SHOW_IM = 0
SHOW_LABELS = 1
SHOW_CONTOURS = 2
# SHOW_FILTERED_LABELS = 3

MODE_VIEWING = 0
MODE_ADDING = 1

class SegViewer(QtGui.QMainWindow):
    """Main class of the programm."""


    # def __init__(self, datap_1=None, datap_2=None, fname_1=None, fname_2=None, disp_smoothed=False, parent=None):
    def __init__(self, datap1=None, datap2=None):

        QtGui.QWidget.__init__(self, parent=None)
        # self.ui = Ui_MainWindow()

        self.setStyleSheet("""QToolTip {
                           background-color: black;
                           color: white;
                           border: black solid 1px
                           }""")

        self.ui = SegViewerGUI()
        self.ui.setupUi(self)

        # uprava stylu pro lepsi vizualizaci splitteru
        QtGui.QApplication.setStyle(QtGui.QStyleFactory.create('Cleanlooks'))

        # self.im = im
        # self.labels = labels
        self.mode = MODE_VIEWING
        self.show_view_L = True
        self.show_view_R = False

        self.view_L = data_view_widget.SliceBox(self)
        self.view_R = data_view_widget.SliceBox(self)

        self.data_1 = Data.Data()
        self.data_2 = Data.Data()

        self.data_L = None
        self.data_R = None

        self.show_mode_L = SHOW_IM
        self.show_mode_R = SHOW_IM

        # load parameters
        self.params = {
            'win_l': 50,
            'win_w': 350,
            'alpha': 4,
            'beta': 1,
            'zoom': 0,
            'scale': 0.25,
            'perc': 30,
            'k_std_h': 3,
            'healthy_simple_estim': 0,
            'prob_w': 0.0001,
            'unaries_as_cdf': 0,
            'bgd_label': 0,
            'hypo_label': 1,
            'healthy_label': 2,
            'hyper_label': 3,
            'data_dir': '/home/tomas/Data/liver_segmentation'
            # 'voxel_size': (1, 1, 1)
        }
        self.params.update(self.load_parameters())

        self.actual_slice_L = 0
        self.actual_slice_R = 0

        self.selected_objects_labels = None  # list of objects' labels selected in tableview

        # self.voxel_size = self.params['voxel_size']
        self.view_widget_width = 50
        self.two_views = False

        if datap1 is not None or datap2 is not None:
            self.setup_data(datap1, datap2)

        # combo boxes - for figure views
        self.ui.figure_L_CB.currentIndexChanged.connect(self.figure_L_CB_callback)
        self.ui.figure_R_CB.currentIndexChanged.connect(self.figure_R_CB_callback)

        data_viewer_layout = QtGui.QHBoxLayout()
        # data_viewer_layout.addWidget(self.form_widget)
        data_viewer_layout.addWidget(self.view_L)
        data_viewer_layout.addWidget(self.view_R)
        self.ui.viewer_F.setLayout(data_viewer_layout)

        # connecting callbacks ----------------------------------
        self.ui.view_L_BTN.clicked.connect(self.view_L_callback)
        self.ui.view_R_BTN.clicked.connect(self.view_R_callback)

        # show image data
        self.ui.show_im_L_BTN.clicked.connect(self.show_im_L_callback)
        self.ui.show_im_R_BTN.clicked.connect(self.show_im_R_callback)

        # show label data
        self.ui.show_labels_L_BTN.clicked.connect(self.show_labels_L_callback)
        self.ui.show_labels_R_BTN.clicked.connect(self.show_labels_R_callback)

        # show contours data
        self.ui.show_contours_L_BTN.clicked.connect(self.show_contours_L_callback)
        self.ui.show_contours_R_BTN.clicked.connect(self.show_contours_R_callback)

        # connecting scrollbars
        self.ui.slice_C_SB.valueChanged.connect(self.slider_C_changed)
        self.ui.slice_L_SB.valueChanged.connect(self.slider_L_changed)
        self.ui.slice_R_SB.valueChanged.connect(self.slider_R_changed)

        # to be able to capture key press events immediately
        self.setFocus()

    def setup_data(self, datap1=None, datap2=None):
        # DATA ---------------------------------------------
        if datap1 is not None:
            self.data_1 = Data.Data()
            self.data_1.create_data(datap1, 'datap1', self.params)
            self.params['voxel_size'] = datap1['voxelsize_mm']
            self.params['voxels2ml_k'] = np.prod(self.params['voxel_size']) * 0.001
        if datap2 is not None:
            self.data_2 = Data.Data()
            self.data_2.create_data(datap2, 'datap2', self.params)

        if self.data_1.loaded:

            #seting up figure and data_L, data_R
            self.ui.figure_L_CB.addItem(self.data_1.filename.split('/')[-1])
            self.ui.figure_R_CB.addItem(self.data_1.filename.split('/')[-1])
            self.data_L = self.data_1
            self.active_data_idx = 1
            self.active_data = self.data_1
            if not self.data_2.loaded:
                self.data_R = self.data_1
                self.ui.slice_R_SB.setMaximum(self.data_1.n_slices - 1)

            self.ui.slice_C_SB.setMaximum(self.data_1.n_slices - 1)
            self.ui.slice_L_SB.setMaximum(self.data_1.n_slices - 1)
            self.ui.show_labels_L_BTN.setEnabled(True)
            self.ui.show_contours_L_BTN.setEnabled(True)

        if self.data_2.loaded:
            self.ui.figure_L_CB.addItem(self.data_2.filename.split('/')[-1])
            self.ui.figure_R_CB.addItem(self.data_2.filename.split('/')[-1])
            self.data_R = self.data_2
            self.ui.figure_R_CB.setCurrentIndex(1)
            if not self.data_1.loaded:
                self.data_L = self.data_2
                self.active_data = self.data_2
                self.active_data_idx = 2

                self.ui.slice_C_SB.setMaximum(self.data_2.n_slices - 1)
                self.ui.slice_L_SB.setMaximum(self.data_2.n_slices - 1)

                self.ui.show_labels_L_BTN.setEnabled(True)
                self.ui.show_contours_L_BTN.setEnabled(True)

            self.ui.slice_R_SB.setMaximum(self.data_2.n_slices - 1)


        if self.data_L is not None:
            self.view_L.setup_widget(self.data_L.data_aview.shape[:-1], self.params['voxel_size'][1:])
            self.view_L.setCW(self.params['win_l'], 'c')
            self.view_L.setCW(self.params['win_w'], 'w')
            self.view_L.setSlice(self.data_L.data_aview[...,0])
            # mouse click signal
            # self.view_L.mouseClickSignal.connect(self.mouse_click_event)
            # self.view_L.mousePressEvent = self.view_L.myMousePressEvent

        if self.data_R is not None:
            self.view_R.setup_widget(self.data_R.data_aview.shape[:-1], self.params['voxel_size'][1:])
            self.view_R.setCW(self.params['win_l'], 'c')
            self.view_R.setCW(self.params['win_w'], 'w')
            self.view_R.setSlice(self.data_R.data_aview[...,0])
            if not self.show_view_L:
                self.view_L.setVisible(False)
            if not self.show_view_R:
                self.view_R.setVisible(False)

    def keyPressEvent(self, QKeyEvent):
        print 'key event: ',
        key = QKeyEvent.key()
        if key == QtCore.Qt.Key_Escape:
            print 'Escape'
            self.close()
        else:
            print key, ' - unrecognized hot key.'


    def load_parameters(self, config_path='config.ini'):
        config = ConfigParser.ConfigParser()
        config.read(config_path)

        params = dict()

        # an automatic way
        for section in config.sections():
            for option in config.options(section):
                try:
                    params[option] = config.getint(section, option)
                except ValueError:
                    try:
                        params[option] = config.getfloat(section, option)
                    except ValueError:
                        if option == 'voxel_size':
                            str = config.get(section, option)
                            params[option] = np.array(map(int, str.split(', ')))
                        else:
                            params[option] = config.get(section, option)

        return params

    def scroll_event(self, value, who):
        if who == 0:  # left viewer
            new = self.actual_slice_L + value
            if (new < 0) or (new >= self.data_L.n_slices):
                return
        elif who == 1:  # right viewer
            new = self.actual_slice_R + value
            if (new < 0) or (new >= self.data_R.n_slices):
                return

    def slider_C_changed(self, val):
        if val == self.actual_slice_L:
            return

        if (val >= 0) and (val < self.data_L.n_slices):
            diff = val - self.actual_slice_L
            self.actual_slice_L = val
        else:
            return

        new_slice_R = self.actual_slice_R + diff
        if new_slice_R < 0:
            new_slice_R = 0
        elif new_slice_R >= self.data_R.n_slices:
            new_slice_R = self.data_R.n_slices - 1

        self.actual_slice_R = new_slice_R

        self.ui.slice_L_SB.setValue(self.actual_slice_L)
        self.ui.slice_R_SB.setValue(self.actual_slice_R)

        im_L = self.get_image('L')
        im_R = self.get_image('R')
        if self.show_mode_L == SHOW_CONTOURS:
            labels_L = self.data_L.labels_filt[self.actual_slice_L, :, :]
            # obj_centers_L = self.data_L.object_centers_filt[self.actual_slice_L, ...]

        else:
            labels_L = None
            # obj_centers_L = None
        if self.show_mode_R == SHOW_CONTOURS:
            labels_R = self.data_R.labels_filt[self.actual_slice_R, :, :]
            # obj_centers_R = self.data_R.object_centers_filt[self.actual_slice_R, ...]
        else:
            labels_R = None
            # obj_centers_R = None

        # self.view_L.setSlice(im_L, contours=labels_L, centers=obj_centers_L)
        # self.view_R.setSlice(im_R, contours=labels_R, centers=obj_centers_R)
        self.view_L.setSlice(im_L, contours=labels_L)
        self.view_R.setSlice(im_R, contours=labels_R)

        self.ui.slice_number_L_LBL.setText('%i/%i' % (self.actual_slice_L + 1, self.data_L.n_slices))
        # self.ui.slice_number_C_LBL.setText('slice # = %i/%i' % (self.actual_slice_L + 1, self.data_L.n_slices))

    def slider_L_changed(self, val):
        if val == self.actual_slice_L:
            return

        if (val >= 0) and (val < self.data_L.n_slices):
            self.actual_slice_L = val
        else:
            return

        self.ui.slice_C_SB.setValue(self.actual_slice_L)

        im_L = self.get_image('L')
        if self.show_mode_L == SHOW_CONTOURS:
            labels_L = self.data_L.labels_filt[self.actual_slice_L, :, :]
            # obj_centers = self.data_L.object_centers_filt[self.actual_slice_L, ...]
            obj_centers = None
        else:
            labels_L = None
            obj_centers = None

        # self.view_L.setSlice(im_L, contours=labels_L, centers=obj_centers)
        self.view_L.setSlice(im_L, contours=labels_L)

        self.ui.slice_number_L_LBL.setText('%i/%i' % (self.actual_slice_L + 1, self.data_L.n_slices))
        # self.ui.slice_number_C_LBL.setText('slice # = %i/%i' % (self.actual_slice_L + 1, self.data_L.n_slices))

    def slider_R_changed(self, val):
        if (val >= 0) and (val < self.data_R.n_slices):
            self.actual_slice_R = val
        else:
            return

        self.ui.slice_number_R_LBL.setText('%i/%i' % (self.actual_slice_R + 1, self.data_R.n_slices))

        im_R = self.get_image('R')
        if self.show_mode_R == SHOW_CONTOURS:
            labels_R = self.data_R.labels_filt[self.actual_slice_R, :, :]
            # obj_centers = self.data_R.object_centers_filt[self.actual_slice_R, ...]
            obj_centers = None
        else:
            labels_R = None
            obj_centers = None

        # self.view_R.setSlice(im_R, contours=labels_R, centers=obj_centers)
        self.view_R.setSlice(im_R, contours=labels_R)


    def view_L_callback(self):
        if self.show_view_L != self.show_view_R:  # logical XOR
            self.one_view_size = self.size()

        self.show_view_L = not self.show_view_L
        self.view_L.setVisible(self.show_view_L)

        # enabling and disabling other toolbar icons
        self.ui.show_im_L_BTN.setEnabled(self.show_view_L)

        # if self.show_view_L and self.data_L.labels is not None:
        if self.show_view_L and self.data_L.processed:
            self.ui.show_labels_L_BTN.setEnabled(True)
            self.ui.show_contours_L_BTN.setEnabled(True)
        else:
            self.ui.show_labels_L_BTN.setEnabled(False)
            self.ui.show_contours_L_BTN.setEnabled(False)

        # self.statusBar().showMessage('Left view set to %s' % self.show_view_L)
        # print 'view_1 set to', self.show_view_1

        self.view_L.update()

        # resizing back to one view size
        if self.show_view_L != self.show_view_R:  # logical XOR
            self.setMinimumSize(self.one_view_size)
            self.resize(self.one_view_size)
            self.setMinimumSize(QtCore.QSize(0, 0))

    def view_R_callback(self):
        if self.show_view_L != self.show_view_R:  # logical XOR
            self.one_view_size = self.size()

        self.show_view_R = not self.show_view_R
        self.view_R.setVisible(self.show_view_R)

        # enabling and disabling other toolbar icons
        self.ui.show_im_R_BTN.setEnabled(not self.ui.show_im_R_BTN.isEnabled())

        if self.show_view_R and self.data_R.processed:
            self.ui.show_labels_R_BTN.setEnabled(True)
            self.ui.show_contours_R_BTN.setEnabled(True)
        else:
            self.ui.show_labels_R_BTN.setEnabled(False)
            self.ui.show_contours_R_BTN.setEnabled(False)

        self.view_R.update()

        # resizing back to one view size
        if self.show_view_L != self.show_view_R:  # logical XOR
            self.setMinimumSize(self.one_view_size)
            self.resize(self.one_view_size)
            self.setMinimumSize(QtCore.QSize(0, 0))

    def show_im_L_callback(self):
        self.show_mode_L = SHOW_IM
        self.view_L.show_mode = self.view_L.SHOW_IM

        im = self.get_image('L')
        self.view_L.setSlice(im)

    def show_im_R_callback(self):
        self.show_mode_R = SHOW_IM
        self.view_R.show_mode = self.view_R.SHOW_IM

        im = self.get_image('R')
        self.view_R.setSlice(im)

    def show_labels_L_callback(self):
        self.show_mode_L = SHOW_LABELS
        self.view_L.show_mode = self.view_L.SHOW_LABELS

        im = self.get_image('L')
        self.view_L.setSlice(im)

    def show_labels_R_callback(self):
        self.show_mode_R = SHOW_LABELS
        self.view_R.show_mode = self.view_R.SHOW_LABELS

        im = self.get_image('R')
        self.view_R.setSlice(im)

    def show_contours_L_callback(self):
        if self.show_mode_L == SHOW_CONTOURS:
            self.view_L.contours_mode_is_fill = not self.view_L.contours_mode_is_fill
        self.show_contours_L()

    def show_contours_L(self):
        self.show_mode_L = SHOW_CONTOURS
        self.view_L.show_mode = self.view_L.SHOW_CONTOURS

        im = self.get_image('L')
        labels = self.data_L.labels_filt[self.actual_slice_L, :, :]
        self.view_L.setSlice(im, contours=labels)

    def show_contours_R_callback(self):
        if self.show_mode_R == SHOW_CONTOURS:
            self.view_R.contours_mode_is_fill = not self.view_R.contours_mode_is_fill
        self.show_contours_R()

    def show_contours_R(self):
        self.show_mode_R = SHOW_CONTOURS
        self.view_R.show_mode = self.view_R.SHOW_CONTOURS

        im = self.get_image('R')
        labels = self.data_R.labels_filt[self.actual_slice_R, :, :]
        self.view_R.setSlice(im, contours=labels)

    def update_view_L(self):
        if self.show_view_L:
            if self.view_L.show_mode == self.view_L.SHOW_LABELS:
                self.show_labels_L_callback()
            elif self.view_L.show_mode == self.view_L.SHOW_CONTOURS:
                self.show_contours_L()

    def update_view_R(self):
        if self.show_view_R:
            if self.view_R.show_mode == self.view_R.SHOW_LABELS:
                self.show_labels_R_callback()
            elif self.view_R.show_mode == self.view_R.SHOW_CONTOURS:
                self.show_contours_R()

    def figure_L_CB_callback(self):
        if self.ui.figure_L_CB.currentIndex() == 0:
            self.data_L = self.data_1
        elif self.ui.figure_L_CB.currentIndex() == 1:
            self.data_L = self.data_2

        if self.actual_slice_L >= self.data_L.n_slices:
            self.actual_slice_L = self.data_L.n_slices - 1

        self.ui.slice_L_SB.setMaximum(self.data_L.n_slices - 1)
        self.ui.slice_C_SB.setMaximum(self.data_L.n_slices - 1)

        if (self.data_L.labels is not None) and self.show_view_L:
            self.ui.show_labels_L_BTN.setEnabled(True)
            self.ui.show_contours_L_BTN.setEnabled(True)
        else:
            self.ui.show_labels_L_BTN.setEnabled(False)
            self.ui.show_contours_L_BTN.setEnabled(False)

        self.view_L.reinit((self.data_L.shape[2], self.data_L.shape[1]))
        self.show_im_L_callback()

    def figure_R_CB_callback(self):
        if self.ui.figure_R_CB.currentIndex() == 0:
            self.data_R = self.data_1
        elif self.ui.figure_R_CB.currentIndex() == 1:
            self.data_R = self.data_2

        if self.actual_slice_R >= self.data_R.n_slices:
            self.actual_slice_R = self.data_R.n_slices - 1

        self.ui.slice_R_SB.setMaximum(self.data_R.n_slices - 1)

        if (self.data_R.labels is not None) and self.show_view_R:
            self.ui.show_labels_R_BTN.setEnabled(True)
            self.ui.show_contours_R_BTN.setEnabled(True)
        else:
            self.ui.show_labels_R_BTN.setEnabled(False)
            self.ui.show_contours_R_BTN.setEnabled(False)

        # self.view_R.reinit(self.data_R.shape[2:0:-1])
        self.view_R.reinit((self.data_R.shape[2], self.data_R.shape[1]))
        self.show_im_R_callback()

    def get_image(self, site):
        im = None
        if site == 'L':
            if self.show_mode_L == SHOW_IM or self.show_mode_L == SHOW_CONTOURS:
                im = self.data_L.data_aview[..., self.actual_slice_L]
            elif self.show_mode_L == SHOW_LABELS:
                # im = self.data_L.labels_aview[...,self.actual_slice_L]
                im = self.data_L.labels_filt_aview[..., self.actual_slice_L]
            # elif self.show_mode_L == SHOW_FILTERED_LABELS:
            #     im = self.data_L.labels_filt_aview[...,self.actual_slice_L]
        elif site == 'R':
            if self.show_mode_R == SHOW_IM or self.show_mode_R == SHOW_CONTOURS:
                im = self.data_R.data_aview[..., self.actual_slice_R]
            elif self.show_mode_R == SHOW_LABELS:
                # im = self.data_R.labels_aview[...,self.actual_slice_R]
                im = self.data_R.labels_filt_aview[..., self.actual_slice_R]
            # elif self.show_mode_R == SHOW_FILTERED_LABELS:
            #     im = self.data_R.labels_filt_aview[...,self.actual_slice_R]
        return im


################################################################################
################################################################################
if __name__ == '__main__':
    #TODO: udelat synteticka data a nevazat se na moje konkretni data
    fname_1 = '/home/tomas/Data/liver_segmentation/seg_rw/seg_rw_183_venous.pklz'
    fname_2 = '/home/tomas/Data/liver_segmentation/seg_he_pipeline/seg_he_pipeline_183_venous.pklz'

    datap_1 = tools.load_pickle_data(fname_1, return_datap=True)
    datap_2 = tools.load_pickle_data(fname_2, return_datap=True)
    # seg = np.load(fname_1)

    # starting application
    app = QtGui.QApplication(sys.argv)
    le = SegViewer(datap1=datap_1, datap2=datap_2)
    le.show()
    sys.exit(app.exec_())
