NUM_CLASSES = 58
NUM_TYPES = 7
HEADER_SIZE = 60

CLASSES = ['AIMchat', 'aim_chat', 'email', 'facebook_audio', 'facebookchat', 'facebook_chat', 'facebook_video', 'ftps_down', 'ftps_up', 'gmailchat', 'hangout_chat', 'hangouts_audio', 'hangouts_chat', 'hangouts_video', 'ICQchat', 'icq_chat', 'netflix', 'scp', 'scpDown', 'scpUp', 'sftp', 'sftpDown', 'sftp_down', 'sftpUp', 'sftp_up', 'skype_audio', 'skype_chat', 'skype_file', 'skype_video', 'spotify', 'torFacebook', 'torGoogle', 'Torrent', 'torTwitter', 'torVimeo', 'torYoutube', 'vimeo', 'voipbuster', 'vpn_aim_chat', 'vpn_bittorrent', 'vpn_email', 'vpn_facebook_audio', 'vpn_facebook_chat', 'vpn_ftps', 'vpn_hangouts_audio', 'vpn_hangouts_chat', 'vpn_icq_chat', 'vpn_netflix', 'vpn_sftp', 'vpn_skype_audio', 'vpn_skype_chat', 'vpn_skype_files', 'vpn_spotify', 'vpn_vimeo', 'vpn_voipbuster', 'vpn_youtube', 'youtube', 'youtubeHTML']

TYPE_NAMES = ['chat', 'email', 'voip', 'streaming', 'file', 'browsing', 'p2p']

# These are the mappings between the low-level labels (as in CLASSES list) and 
# high-level ones (as in type_names list)
TYPE_MAP = [0, 0, 1, 2, 0, 0, 3, 4, 4, 0, 0, 2, 0, 3, 0, 0, 3, 4, 4, 4, 4, 4, 4, 4, 4, 
            2, 0, 4, 3, 3, 5, 5, 6, 5, 5, 3, 3, 2, 0, 6, 1, 3, 0, 4, 2, 0, 0, 3, 4, 2,
            0, 4, 3, 3, 2, 3, 3, 5]

MASKED_VAL = -1