import numpy as np
from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QLayout
from Orange.data import Table, ContinuousVariable, Domain, StringVariable
from Orange.widgets.gui import widgetBox, widgetLabel, comboBox, auto_commit
from Orange.widgets.settings import Setting
from Orange.widgets.utils.itemmodels import VariableListModel
from Orange.widgets.widget import OWWidget, Default

from orangecontrib.imageanalytics.face_embedder import FaceEmbedder


class _Input:
    IMAGES = 'Images'
    REFERENCE = 'Reference Image'


class _Output:
    EMBEDDINGS = 'Embeddings'
    SKIPPED_IMAGES = 'Skipped Images'


class OWFaceCompare(OWWidget):
    # todo: implement embedding in a non-blocking manner
    # todo: stop running task action
    name = "Face Comparator"
    description = "Image embedding through deep neural networks."
    icon = "icons/Face.svg"
    priority = 150

    want_main_area = False
    _auto_apply = Setting(default=True)

    inputs = [(_Input.IMAGES, Table, 'set_data'),
            (_Input.REFERENCE, Table, 'set_reference')]
    outputs = [
        (_Output.EMBEDDINGS, Table, Default),
        (_Output.SKIPPED_IMAGES, Table)
    ]

    cb_image_attr_current_id = Setting(default=0)
    cb_image_attr_current_id_ref = Setting(default=0)
    _NO_DATA_INFO_TEXT = "No data on input."

    def __init__(self):
        super().__init__()
        self._image_attributes = None
        self._image_attributes_ref = None
        self._input_data = None
        self._input_reference = None
        
        self._setup_layout()

        self._image_embedder = FaceEmbedder(
            model='compare-v1',
            layer='penultimate',
        )
        self._set_server_info(
            self._image_embedder.is_connected_to_server()
        )

    def _setup_layout(self):
        self.controlArea.setMinimumWidth(self.controlArea.sizeHint().width())
        self.layout().setSizeConstraint(QLayout.SetFixedSize)

        widget_box = widgetBox(self.controlArea, 'Info')
        self.input_data_info = widgetLabel(widget_box, self._NO_DATA_INFO_TEXT)
        self.connection_info = widgetLabel(widget_box, "")

        widget_box = widgetBox(self.controlArea, 'Settings')
        self.cb_image_attr = comboBox(
            widget=widget_box,
            master=self,
            value='cb_image_attr_current_id',
            label='Image attribute:',
            orientation=Qt.Horizontal,
            callback=self._cb_image_attr_changed
        )
        """
        self.cb_image_attr = comboBox(
            widget=widget_box,
            master=self,
            value='cb_image_attr_current_id_ref',
            label='Image attribute (reference):',
            orientation=Qt.Horizontal,
            callback=self._cb_image_attr_changed
        )
        """
        self.auto_commit_widget = auto_commit(
            widget=self.controlArea,
            master=self,
            value='_auto_apply',
            label='Apply',
            checkbox_label='Auto Apply',
            commit=self.commit
        )

    def set_data(self, data):
        if data is None:
            self.send(_Output.EMBEDDINGS, None)
            self.send(_Output.SKIPPED_IMAGES, None)
            self.input_data_info.setText(self._NO_DATA_INFO_TEXT)
            return

        self._image_attributes = self._filter_image_attributes(data)
        if not self._image_attributes:
            input_data_info_text = (
                "Data with {:d} instances, but without image attributes."
                .format(len(data)))
            input_data_info_text.format(input_data_info_text)
            self.input_data_info.setText(input_data_info_text)
            self._input_data = None
            return

        if not self.cb_image_attr_current_id < len(self._image_attributes):
            self.cb_image_attr_current_id = 0

        self.cb_image_attr.setModel(VariableListModel(self._image_attributes))
        self.cb_image_attr.setCurrentIndex(self.cb_image_attr_current_id)

        self._input_data = data
        input_data_info_text = "Data with {:d} instances.".format(len(data))
        self.input_data_info.setText(input_data_info_text)

        self._cb_image_attr_changed()

    def set_reference(self, data):
        if data is None:
            self.send(_Output.EMBEDDINGS, None)
            self.send(_Output.SKIPPED_IMAGES, None)
            self.input_data_info.setText(self._NO_DATA_INFO_TEXT)
            return

        self._image_attributes_ref = self._filter_image_attributes(data)
        if not self._image_attributes_ref:
            input_data_info_text = (
                "Data with {:d} instances, but without image attributes."
                .format(len(data)))
            input_data_info_text.format(input_data_info_text)
            self.input_data_info.setText(input_data_info_text)
            self._input_data = None
            return

        if not self.cb_image_attr_current_id_ref < len(self._image_attributes_ref):
            self.cb_image_attr_current_id_ref = 0

        self.cb_image_attr.setModel(VariableListModel(self._image_attributes_ref))
        self.cb_image_attr.setCurrentIndex(self.cb_image_attr_current_id)

        self._input_reference = data
        input_data_info_text = "Data with {:d} instances.".format(len(data))
        self.input_data_info.setText(input_data_info_text)

        self._cb_image_attr_changed()

    @staticmethod
    def _filter_image_attributes(data):
        metas = data.domain.metas
        return [m for m in metas if m.attributes.get('type') == 'image']

    def _cb_image_attr_changed(self):
        if self._auto_apply:
            self.commit()

    def commit(self):
        if not self._image_attributes or not self._input_data or not self._input_reference:
            self.send(_Output.EMBEDDINGS, None)
            self.send(_Output.SKIPPED_IMAGES, None)
            return

        self.auto_commit_widget.setDisabled(True)

        file_paths_attr = self._image_attributes[self.cb_image_attr_current_id]
        file_paths = self._input_data[:, file_paths_attr].metas.flatten()
        
        file_paths_attr_ref = self._image_attributes_ref[self.cb_image_attr_current_id_ref]
        file_paths_ref = self._input_reference[:, file_paths_attr_ref].metas.flatten()

        with self.progressBar(len(file_paths)) as progress:
            try:
                embeddings = self._image_embedder(
                    file_paths=file_paths,
                    file_paths_ref=file_paths_ref,
                    image_processed_callback=lambda: progress.advance()
                )
            except ConnectionError:
                self.send(_Output.EMBEDDINGS, None)
                self.send(_Output.SKIPPED_IMAGES, None)
                self._set_server_info(connected=False)
                self.auto_commit_widget.setDisabled(False)
                return

        self._send_output_signals(embeddings)
        self.auto_commit_widget.setDisabled(False)

    def _send_output_signals(self, embeddings):
        skipped_images_bool = np.array([x is None for x in embeddings])

        if np.any(skipped_images_bool):
            skipped_images = self._input_data[skipped_images_bool]
            skipped_images = Table(skipped_images)
            skipped_images.ids = self._input_data.ids[skipped_images_bool]
            self.send(_Output.SKIPPED_IMAGES, skipped_images)
        else:
            self.send(_Output.SKIPPED_IMAGES, None)

        embedded_images_bool = np.logical_not(skipped_images_bool)

        if np.any(embedded_images_bool):
            embedded_images = self._input_data[embedded_images_bool]

            embeddings = embeddings[embedded_images_bool]
            embeddings = np.stack(embeddings)

            embedded_images = self._construct_output_data_table(
                embedded_images,
                embeddings
            )
            embedded_images.ids = self._input_data.ids[embedded_images_bool]
            self.send(_Output.EMBEDDINGS, embedded_images)
        else:
            self.send(_Output.EMBEDDINGS, None)

    @staticmethod
    def _construct_output_data_table(embedded_images, embeddings):
        dimensions = range(embeddings.shape[1])
        attributes = list(embedded_images.domain.attributes)
        class_vars = [ContinuousVariable('similarity')]
        new_metas = [StringVariable('match'), StringVariable('match image'), StringVariable('Label')]
        
        X = embedded_images.X
        Y = np.array([e[1] for e in embeddings]).reshape(-1, 1)
        
        M = np.array([[e[0], e[2], e[0]] for e in embeddings])
        metas = np.hstack((embedded_images.metas, M))
        
        domain = Domain(
            attributes=attributes,
            class_vars=class_vars,
            metas=list(embedded_images.domain.metas) + new_metas
        )
        neighbors_img_attr = None
        neighbors_img_name = None
        reference_img_attr = None
        reference_img_name = None
        label_name = None
        
        for i, attr in enumerate(domain.metas):
            if attr.name == 'match image':
                neighbors_img_attr = i
            if attr.name == 'match':
                neighbors_img_name = i
            if attr.name == 'image':
                reference_img_attr = i
            if attr.name == 'image name' or attr.name == 'name':
                reference_img_name = i
            if attr.name == 'Label':
                label_name = i
        
        x = metas[:,neighbors_img_name].copy()
        metas[:,neighbors_img_name] = metas[:,reference_img_name]
        metas[:,reference_img_name] = x
        metas[:,label_name] = metas[:,neighbors_img_name]
        
        x = metas[:,neighbors_img_attr].copy()
        metas[:,neighbors_img_attr] = metas[:,reference_img_attr]
        metas[:,reference_img_attr] = x
        return Table(domain, X, Y, metas)

    def _set_server_info(self, connected):
        self.clear_messages()
        if connected:
            self.connection_info.setText("Connected to server.")
        else:
            self.connection_info.setText("")
            self.warning("Not connected to server.")

    def onDeleteWidget(self):
        super().onDeleteWidget()
        self._image_embedder.__exit__(None, None, None)


if __name__ == '__main__':
    import sys
    from AnyQt.QtWidgets import QApplication
    app = QApplication(sys.argv)
    widget = OWFace()
    widget.show()
    app.exec()
    widget.saveSettings()
