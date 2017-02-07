from io import BytesIO
from itertools import islice
from os.path import join, isfile
import pickle

import numpy as np
from Orange.misc.environ import cache_dir
from PIL.Image import open as open_image, LANCZOS

from orangecontrib.imageanalytics.http2_client import Http2Client
from orangecontrib.imageanalytics.http2_client import MaxNumberOfRequestsError
from orangecontrib.imageanalytics.utils import md5_hash
from orangecontrib.imageanalytics.utils import save_pickle, load_pickle

from orangecontrib.imageanalytics.image_embedder import ImageEmbedder

MODELS_SETTINGS = {
    'inception-v3': {
        'target_image_size': (299, 299),
        'layers': ['penultimate']
    },
    'compare-v1': {
        'target_image_size': (299, 299),
        'layers': ['penultimate']
    },
    'classify-v1': {
        'target_image_size': (299, 299),
        'layers': ['penultimate']
    },
}


class FaceEmbedder(ImageEmbedder):
    """"Client side functionality for accessing a remote http2
    image embedding backend.

    Examples
    --------
    >>> from orangecontrib.imageanalytics.image_embedder import ImageEmbedder
    >>> image_file_paths = [...]
    >>> with ImageEmbedder(model='model_name', layer='penultimate') as embedder:
    ...    embeddings = embedder(image_file_paths)
    """
    _cache_file_blueprint = '{:s}_{:s}_embeddings.pickle'

    def __init__(self, model, layer,
                 server_url='193.2.72.50', server_port=8080):
        super().__init__(model, layer, server_url=server_url, server_port=server_port)
        model_settings = self._get_model_settings_confidently(model, layer)
        
        self._model = model
        self._layer = layer
        self._target_image_size = model_settings['target_image_size']

        cache_file_path = self._cache_file_blueprint.format(model, layer)
        self._cache_file_path = join(cache_dir(), cache_file_path)
        self._cache_dict = self._init_cache()

    @staticmethod
    def _get_model_settings_confidently(model, layer):
        if model not in MODELS_SETTINGS.keys():
            model_error = "'{:s}' is not a valid model, should be one of: {:s}"
            available_models = ', '.join(MODELS_SETTINGS.keys())
            raise ValueError(model_error.format(model, available_models))

        model_settings = MODELS_SETTINGS[model]

        if layer not in model_settings['layers']:
            layer_error = (
                "'{:s}' is not a valid layer for the '{:s}'"
                " model, should be one of: {:s}")
            available_layers = ', '.join(model_settings['layers'])
            raise ValueError(layer_error.format(layer, model, available_layers))

        return model_settings
    
    def _send_to_server(self, file_paths, ref_path, image_processed_callback):
        """ Load images and compute cache keys and send requests to
        an http2 server for valid ones.
        """
        cache_keys = []
        http_streams = []

        for file_path in file_paths:
            image = self._load_image_or_none(file_path)
            dct = {'image': image, 'action': 'classify'}
            body_bytes = pickle.dumps(dct, protocol=2)
            
            if not image:
                # skip the sending because image was skipped at loading
                http_streams.append(None)
                cache_keys.append(None)
                continue

            cache_key = md5_hash(image)
            cache_keys.append(cache_key)
            #if cache_key in self._cache_dict:
                # skip the sending because image is present in the
                # local cache
                #http_streams.append(None)
                #continue

            try:
                headers = {
                    'Content-Type': 'image/jpeg',
                    'Content-Length': str(len(body_bytes))
                }
                
                """
                if self._model == 'inception-v3':
                    stream_id = self._send_request(
                        method='POST',
                        url='/images/' + self._model,
                        headers=headers,
                        body_bytes=image
                    )
                elif self._model == 'compare-v1':
                    ref_image = self._load_image_or_none(ref_path)
                    dct = {'image': image, 'ref': ref_image, 'ref_path': ref_path, 'img_path': file_path} 
                    body_bytes = pickle.dumps(dct, protocol=2)
                    headers['Content-Length'] = str(len(body_bytes))
                    stream_id = self._send_request(
                        method='POST',
                        url='/compare/' + self._model,
                        headers=headers,
                        body_bytes=body_bytes
                    )
                """
                if self._model == 'classify-v1':
                    stream_id = self._send_request(
                        method='POST',
                        url='/classify/' + self._model,
                        headers=headers,
                        body_bytes=body_bytes
                    )
                http_streams.append(stream_id)
            except ConnectionError:
                self.persist_cache()
                raise
        # wait for the responses in a blocking manner
        return self._get_responses_from_server(
            http_streams,
            cache_keys,
            image_processed_callback
        )

    def _get_responses_from_server(self, http_streams, cache_keys,
                                   image_processed_callback):
        """Wait for responses from an http2 server in a blocking manner."""
        embeddings = []
        for stream_id, cache_key in zip(http_streams, cache_keys):
            if not stream_id:
                # skip rest of the waiting because image was either
                # skipped at loading or is present in the local cache
                embedding = self._get_cached_result_or_none(cache_key)
                embeddings.append(embedding)

                if image_processed_callback:
                    image_processed_callback()
                continue

            try:
                response = self._get_json_response_or_none(stream_id)
            except ConnectionError:
                self.persist_cache()
                raise

            if not response or 'match' not in response:
                # returned response is not a valid json response
                # or the embedding key not present in the json
                embeddings.append(None)
            else:
                # successful response
                t = [response['match'], float(response['score']), response['image']]
                embeddings.append(t)
                self._cache_dict[cache_key] = t

            if image_processed_callback:
                image_processed_callback()
        
        return embeddings
        
    def __call__(self, file_paths, file_paths_ref=None, image_processed_callback=None):
        """Send the images to the remote server in batches. The batch size
        parameter is set by the http2 remote peer (i.e. the server).

        Parameters
        ----------
        file_paths: list
            A list of file paths for images to be embedded.

        image_processed_callback: callable (default=None)
            A function that is called after each image is fully processed
            by either getting a successful response from the server,
            getting the result from cache or skipping the image.

        Returns
        -------
        embeddings: array-like
            Array-like of float16 arrays (embeddings) for
            successfully embedded images and Nones for skipped images .

        Raises
        ------
        ConnectionError:
            If disconnected or connection with the server is lost
            during the embedding process.
        """
        # check connection at the beginning to avoid doing unnecessary work
        if not self.is_connected_to_server():
            reconnected = self.reconnect_to_server()

            if not reconnected:
                raise ConnectionError(self._conn_err_msg)

        ref_image = None
        if file_paths_ref != None:
            ref_image = file_paths_ref[0]

        all_embeddings = []
        gen = self._yield_in_batches(file_paths)
        for batch in iter(gen):
            try:
                embeddings = self._send_to_server(
                    batch, ref_image, image_processed_callback
                )
            except MaxNumberOfRequestsError:
                # maximum number of http2 requests through a single
                # connection is exceeded and a remote peer has closed
                # the connection so establish a new connection and retry
                # with the same batch (should happen rarely as the setting
                # is usually set to >= 1000 requests in http2)
                self.reconnect_to_server()
                embeddings = self._send_to_server(
                    batch, ref_image, image_processed_callback
                )

            all_embeddings += embeddings
            self.persist_cache()

        return np.array(all_embeddings)