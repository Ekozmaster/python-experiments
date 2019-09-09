class DisplayResolutions:
    # Square
    s_64 = (64, 64)
    s_128 = (128, 128)
    s_256 = (256, 256)
    s_512 = (512, 512)
    s_1024 = (1024, 1024)

    # Height for 16:9 aspect ratio
    h_720 = (1280, 720)  # HD
    h_1080 = (1920, 1080)  # FHD
    h_1440 = (2560, 1440)  # QHD
    h_2160 = (3840, 2160)  # 4K
    h_4320 = (7680, 4320)  # 8K

    # Get multiples of 16x9, 4x3, etc. (Standard resolutions' ratios. eg: m_16x9(120) = FHD)
    @staticmethod
    def m_16x9(factor):
        return 16 * factor, 9 * factor

    @staticmethod
    def m_4x3(factor):
        return 4 * factor, 3 * factor

    @staticmethod
    def m_square(factor):
        return 1 * factor, 1 * factor