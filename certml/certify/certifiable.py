"""Certifiable Defense"""


class CertifiableMixin:
    """Certifiable Defense"""

    def __init__(self):
        pass

    def cert_params(self):
        raise NotImplementedError('Certification parameters not implemented!')
