import re
import random
import sys
import os
from collections import Counter

RE_VIDEO = r".*(xx\.fbcdn|xhcdn|t8cdn|phncdn|xvideos-cdn|xhcdn.com|xnxx-cdn|ypncdn|t8cdn|sb-cd.com|dailymotion|googlevideo\.com|rdtcdn|nflxvideo|vod.*akamaized|ttvnw.net|vid.*cdn|video.twimg|vod.*cdn|bcrncdn|dditscdn|streaming.estat|cdn.*vid|wlmediahub)"


def is_video(x):
    return re.match(RE_VIDEO, x)

def read_domain_list(name):
    if not isinstance(name, list):
        name = [name]
    lines = []
    for n in name:
        if(os.path.isfile("/home/orange/domains/homemade_dataset/domainsTest/{}".format(n))):
            with open("/home/orange/domains/homemade_dataset/domainsTest/{}".format(n)) as f:
                lines += [line.rstrip() for line in f]
        else:
            #This path is for the modified ut1 dataset.
            #To get the original, please comment this and uncomment the second one
            with open("/home/orange/domains/homemade_dataset/ut_capitole_modified/{}/domains".format(n)) as f:
            #with open("/home/orange/domains/ut_capitole/{}/domains".format(n)) as f:
                lines += [line.rstrip() for line in f]


    lines = sorted(lines, key=lambda x:-len(x))
    
    return lines


def read_domain_list_newDownload(name):
    if not isinstance(name, list):
        name = [name]
    lines = []
    for n in name:
        if(os.path.isfile("/home/orange/domains/homemade_dataset/domainsNewDownload/{}".format(n))):
            with open("/home/orange/domains/homemade_dataset/domainsNewDownload/{}".format(n)) as f:
                lines += [line.rstrip() for line in f]
        else:
            #This path is for the modified ut1 dataset.
            #To get the original, please comment this and uncomment the next
            #with open("/home/orange/domains/homemade_dataset/ut_capitole_modified/{}/domains".format(n)) as f:
            with open("/home/orange/domains/ut_capitole/{}/domains".format(n)) as f:
                lines += [line.rstrip() for line in f]

    #domains = open("/home/orange/domains/homemade_dataset/domainslist","a")
    #for line in lines : domains.write(line+"\n")
    #domains.close
    lines = sorted(lines, key=lambda x:-len(x))
    
    return lines


def is_in_domain(addr, domains):
    return any(map(lambda x:addr.endswith(x), domains))
def is_in_domain2(addr, domains):
    return addr in domains

def longest_matching_domain(addr, domains):
    for d in domains:
    
        #Quick&dirty trick to reduce the complexity
        if("*" in d or "^" in d):
            match = re.match(".*"+d,addr)
            lenMatch = len(match.group(0)) if not(match is None) else 0  
            if (lenMatch > 0):
                return lenMatch
    
        elif (addr.endswith(d)):
            return len(d)
          
            
    return -1

def longest_matching_domain_diag(addr, domains):
    

    for d in domains:
        
        if("*" in d or "^" in d) :
            if(re.match(".*"+d,addr)):
                return("Regex matched addr: "+ addr + " to domain: " +d)
            
        else :

            if(addr.endswith(d) and (not addr==d)):
                return("Matched addr: " + addr + " to domain: " + d)

            if (addr==d):

                return("Exact match found on: " + d) 

    return -1


def longest_matching_domain_regex(addr, domainsCompiled):

    for i in range (len(domainsCompiled)):
        
        match = domainsCompiled[i].match(addr)
        lenMatch = len(match.group(0)) if not(match is None) else 0
        
        #lenMatch = len(domainsCompiled[i].match(addr).group(0)) if not domainsCompiled[i].match(addr) is None else 0
        
        if (lenMatch > 0):
            return lenMatch
    
    return -1


LABELS = ['video', 'browsing', 'social', 'download', '']

def counts_to_weights(counts, weight_cap=None):
    """
    Input: [(class, count)]
    Output: { class: weight }
    """
    
    max_count = max([c for x,c in counts])
    result = dict()
    for x,c in counts:
        if c <= 0:
            raise Exception("Class {} has count {} <= 0".format(x, c))
        result[x] = max_count / c
        if weight_cap:
            result[x] = min(weight_cap, result[x])
    return result


def domain_suffix(x):
    return '.'.join(x.split('.')[-2:])



class Ensemble:
    def __init__(self, names: list, counts: list, weights: list, lfunc, cached=True):
        
        if cached:
            self.cache = dict()
        else:
            self.cache = None
        
        self.names = names
        self.weights = weights
        self.counts = counts
        self._label_func_int = lfunc
        
    def label_func(self, x):
        if self.cache is None:
            return self._label_func_int(x)
        elif x in self.cache:
            return self.cache[x]
        else:
            result = self._label_func_int(x)
            self.cache[x] = result
            return result
        
    def class_label_func(self, cl):
        index = self.names.index(cl)
        
        def f(x):
            if self.label_func(x) == index:
                return 1
            else:
                return 0
            
        return f
    def class_partial_weight(self, cl):
        index = self.names.index(cl)
        
        cl_count = self.counts[index]
        cl_complement = sum(self.counts) - cl_count
        w = cl_complement / cl_count
        return {0:1, 1:w}
    
    def __len__(self):
        return len(self.names)
    
    def total(self):
        return sum(self.counts)
    
    def generate_counts(self, li, weight_cap=None):
        """
        li: list of labels generated by self.label_func
        """
        counts_map = dict(Counter(li))
        self.counts = [counts_map[i] for i in range(len(self.names))]
        self.weights = counts_to_weights(list(enumerate(self.counts)), weight_cap)
        
    

    @staticmethod
    def get_0214_ensemble(cached=True):
        """
        The ensemble is generated from orange TLS dataset. It is used in the multi-class model on Feb. 14.
        """
        domains_socialmedia = read_domain_list("social_networks")
        domains_mail        = read_domain_list(["mail", "webmail"])
        domains_chat        = read_domain_list("chat")
        domains_browsing    = read_domain_list(["forums", "blog", "press", "sports", "cooking", "cleaning"])
        domains_shopping    = read_domain_list(["shopping", "lingerie"])
        domains_ads         = read_domain_list("ads")
        domains_redir       = read_domain_list(["shortener", "redirector", "strong_redirector", "strict_redirector"])

        def cat_video(x):
            return is_video(x)
        def cat_social(x):
            return is_in_domain(x, domains_socialmedia)
        def cat_mail(x):
            return is_in_domain(x, domains_mail)

        def cat_chat(x):
            return is_in_domain(x, domains_chat)
        def cat_browsing(x):
            return is_in_domain(x, domains_browsing)
        def cat_shopping(x):
            return is_in_domain(x, domains_shopping)
        def cat_ads(x):
            return is_in_domain(x, domains_ads)

        def cat_search(x):
            SITES = ["yandex.ru", "yastatic.net", "baidu.com", "bdstatic.com", "duckduckgo.com", "ask.com", "bing.com", \
                     "google.com", "google.fr", "google.ca", \
                     "www.google.com", "www.google.fr", "www.google.ca"]
            SITES = set(SITES)
            return x in SITES

        # Numbers from orange dataset

        li = [
            ("video",    cat_video,     8768),
            ("social",   cat_social,   34128),
            ("mail",     cat_mail,     12340),
            ("chat",     cat_chat,      6663),
            ("browsing", cat_browsing, 34040),
            ("shopping", cat_shopping,  6885),
            ("ads",      cat_ads,      47254),
            ("search",   cat_search,    9043),
        ]
        total = 343229
        
        names = ["other"] + [x[0] for x in li]
        counts = [x[2] for x in li]
        counts = [total-sum(counts)] + counts
        weights = counts_to_weights(list(enumerate(counts)))
        
        def f(x):
            for i,cat in enumerate(li):
                if cat[1](x):
                    return i+1
            return 0
        
        return Ensemble(names, counts, weights, f, cached=cached)



    @staticmethod
    def get_0216_ensemble(cached=True):
        NAMES = ["video", "social", "mail", "download", "chat", "browsing", "shopping", "ads", "search"]
        ENSEMBLE = [
            ("social", read_domain_list("social_networks")),
            ("mail", read_domain_list(["mail", "webmail"])),
            ("download", read_domain_list(["update", "download", "filehosting"])),
            ("chat", read_domain_list("chat")),
            ("browsing", read_domain_list(["forums", "blog", "press", "sports", "cooking", "cleaning"])),
            ("shopping", read_domain_list(["shopping", "lingerie"])),
            ("ads", read_domain_list("ads")),
        ]

        def cat_search(x):
            SITES = ["yandex.ru", "yastatic.net", "baidu.com", "bdstatic.com", "duckduckgo.com", "ask.com", "bing.com", \
                     "google.com", "google.fr", "google.ca", \
                     "www.google.com", "www.google.fr", "www.google.ca"]
            SITES = set(SITES)
            if x in SITES:
                return len(x)
            else:
                return -1


        def f1(x):
            if is_video(x):
                return "video"

            maxmatchs = [(name, longest_matching_domain(x, dom)) for name,dom in ENSEMBLE] \
                + [("search", cat_search(x))]
            name,count = max(maxmatchs, key=lambda x:x[1])

            if count == -1:
                return "unknown"
            else:
                return name
        def f2(x):
            y = f1(x)
            if y == 'unknown':
                return -1
            else:
                return NAMES.index(y)
        
        counts_map = {'social': 32715,
             'browsing': 33971,
             'ads': 50088,
             'search': 9043,
             'mail': 12337,
             'shopping': 6801,
             'chat': 5398,
             'video': 8768,
             'download': 1151}
        counts = [counts_map[x] for x in NAMES]
        weights = counts_to_weights(list(enumerate(counts)))
        
        return Ensemble(NAMES, counts, weights, f2, cached=cached)


    @staticmethod
    def get_0219_ensemble(cached=True):
        """
        Removed fbcdn
        """
        NAMES = ["video", "social", "mail", "download", "chat", "browsing", "shopping", "search"]
        RE_VIDEO2 = r".*(xhcdn|t8cdn|phncdn|xvideos-cdn|xhcdn.com|xnxx-cdn|ypncdn|t8cdn|sb-cd.com|dailymotion|googlevideo\.com|rdtcdn|nflxvideo|vod.*akamaized|ttvnw.net|vid.*cdn|video.twimg|vod.*cdn|bcrncdn|dditscdn|streaming.estat|cdn.*vid|wlmediahub)"
        
        RE_REMOVED = r".*(xx\.fbcdn)"
        
        ENSEMBLE = [
            ("social", read_domain_list("social_networks")),
            ("mail", read_domain_list(["mail", "webmail"])),
            ("download", read_domain_list(["update", "download", "filehosting"])),
            ("chat", read_domain_list("chat")),
            ("browsing", read_domain_list(["forums", "blog", "press", "sports", "cooking", "cleaning"])),
            ("shopping", read_domain_list(["shopping", "lingerie"])),
        ]

        def cat_search(x):
            SITES = ["yandex.ru", "yastatic.net", "baidu.com", "bdstatic.com", "duckduckgo.com", "ask.com", "bing.com", \
                     "google.com", "google.fr", "google.ca", \
                     "www.google.com", "www.google.fr", "www.google.ca"]
            SITES = set(SITES)
            if x in SITES:
                return len(x)
            else:
                return -1


        def f1(x):
            if re.match(RE_VIDEO2, x):
                return "video"
            if re.match(RE_REMOVED, x):
                return "unknown"

            maxmatchs = [(name, longest_matching_domain(x, dom)) for name,dom in ENSEMBLE] \
                + [("search", cat_search(x))]
            name,count = max(maxmatchs, key=lambda x:x[1])

            if count == -1:
                return "unknown"
            else:
                return name
        def f2(x):
            y = f1(x)
            if y == 'unknown':
                return -1
            else:
                return NAMES.index(y)
        
        counts_map = {'social': 32715,
             'browsing': 33971,
             'search': 9043,
             'mail': 12337,
             'shopping': 6801,
             'chat': 5398,
             'video': 8768,
             'download': 1151}
        counts = [counts_map[x] for x in NAMES]
        weights = counts_to_weights(list(enumerate(counts)))
        
        return Ensemble(NAMES, counts, weights, f2, cached=cached)

    @staticmethod
    def get_0221_ensemble(cached=True):
        """
        Removed fbcdn
        """
        NAMES = ["video","social", "mail", "download", "chat", "browsing-shopping", "search"]
        RE_VIDEO2 = r".*(xhcdn|t8cdn|phncdn|xvideos-cdn|xhcdn.com|xnxx-cdn|ypncdn|t8cdn|sb-cd.com|dailymotion|googlevideo\.com|rdtcdn|nflxvideo|vod.*akamaized|ttvnw.net|vid.*cdn|video.twimg|vod.*cdn|bcrncdn|dditscdn|streaming.estat|cdn.*vid|wlmediahub)"
        
        RE_REMOVED = r".*(xx\.fbcdn)"
        
        ENSEMBLE = [
            ("social", read_domain_list("social_networks")),
            ("mail", read_domain_list(["mail", "webmail"])),
            ("download", read_domain_list("download")),
            ("chat", read_domain_list("chat")),
            ("browsing-shopping", read_domain_list(["forums", "blog", "press", "sports", "cooking", "cleaning", \
                                                    "shopping", "lingerie"])),
        ]

        def cat_search(x):
            SITES = ["yandex.ru", "yastatic.net", "baidu.com", "bdstatic.com", "duckduckgo.com", "ask.com", "bing.com", \
                     "google.com", "google.fr", "google.ca", \
                     "www.google.com", "www.google.fr", "www.google.ca"]
            SITES = set(SITES)
            if x in SITES:
                return len(x)
            else:
                return -1


        def f1(x):
            
            if re.match(RE_VIDEO2, x):
                return "video"
            if re.match(RE_REMOVED, x):
                return "unknown"
            
            maxmatchs = [(name, longest_matching_domain(x, dom)) for name,dom in ENSEMBLE] \
                + [("search", cat_search(x))]
            name,count = max(maxmatchs, key=lambda x:x[1])

            if count == -1:
                return "unknown"
            else:
                return name
        def f2(x):
            y = f1(x)
            if y == 'unknown':
                return -1
            else:
                return NAMES.index(y)
        
        
        counts_map = {'social': 32715,
              'video': 8768,
             'browsing-shopping': 33971+6801,
             'search': 9043,
             'mail': 12337,
             'chat': 5398,
             'download': 1151}
        counts = [counts_map[x] for x in NAMES]
        weights = counts_to_weights(list(enumerate(counts)))
        
        return Ensemble(NAMES, counts, weights, f2, cached=cached)

    
    @staticmethod
    def get_0221NewDownload_ensemble(cached=True):
        """
        Removed fbcdn
        """
        NAMES = ["video","social", "mail", "download", "chat", "browsing-shopping", "search"]
        RE_VIDEO2 = r".*(xhcdn|t8cdn|phncdn|xvideos-cdn|xhcdn.com|xnxx-cdn|ypncdn|t8cdn|sb-cd.com|dailymotion|googlevideo\.com|rdtcdn|nflxvideo|vod.*akamaized|ttvnw.net|vid.*cdn|video.twimg|vod.*cdn|bcrncdn|dditscdn|streaming.estat|cdn.*vid|wlmediahub)"
        
        RE_REMOVED = r".*(xx\.fbcdn)"
        
        ENSEMBLE = [
            ("social", read_domain_list_newDownload("social_networks")),
            ("mail", read_domain_list_newDownload(["mail", "webmail"])),
            ("download", read_domain_list_newDownload("download")),
            ("chat", read_domain_list_newDownload("chat")),
            ("browsing-shopping", read_domain_list_newDownload(["forums", "blog", "press", "sports", "cooking", "cleaning", \
                                                    "shopping", "lingerie"])),
        ]

        def cat_search(x):
            SITES = ["yandex.ru", "yastatic.net", "baidu.com", "bdstatic.com", "duckduckgo.com", "ask.com", "bing.com", \
                     "google.com", "google.fr", "google.ca", \
                     "www.google.com", "www.google.fr", "www.google.ca"]
            SITES = set(SITES)
            if x in SITES:
                return len(x)
            else:
                return -1


        def f1(x):
            
            if re.match(RE_VIDEO2, x):
                return "video"
            if re.match(RE_REMOVED, x):
                return "unknown"
            
            maxmatchs = [(name, longest_matching_domain(x, dom)) for name,dom in ENSEMBLE] \
                + [("search", cat_search(x))]
            name,count = max(maxmatchs, key=lambda x:x[1])

            if count == -1:
                return "unknown"
            else:
                return name
        def f2(x):
            y = f1(x)
            if y == 'unknown':
                return -1
            else:
                return NAMES.index(y)
        
        
        counts_map = {'social': 32715,
              'video': 8768,
             'browsing-shopping': 33971+6801,
             'search': 9043,
             'mail': 12337,
             'chat': 5398,
             'download': 1151}
        counts = [counts_map[x] for x in NAMES]
        weights = counts_to_weights(list(enumerate(counts)))
        
        return Ensemble(NAMES, counts, weights, f2, cached=cached)
    
    
    @staticmethod
    def get_0221Modified_ensemble(cached=True):
        """
        Removed fbcdn
        """
        #"video"
        #,
        NAMES = ["social", "mail", "download", "chat", "browsing-shopping", "search"]
        #RE_VIDEO2 = r".*(xhcdn|t8cdn|phncdn|xvideos-cdn|xhcdn.com|xnxx-cdn|ypncdn|t8cdn|sb-cd.com|dailymotion|googlevideo\.com|rdtcdn|nflxvideo|vod.*akamaized|ttvnw.net|vid.*cdn|video.twimg|vod.*cdn|bcrncdn|dditscdn|streaming.estat|cdn.*vid|wlmediahub)"
        
        RE_REMOVED = r".*(xx\.fbcdn)"
        
        ENSEMBLE = [
            
            ("social", read_domain_list("social_networks")),
            ("mail", read_domain_list(["mail", "webmail"])),
            ("download", read_domain_list(["update", "download", "filehosting"])),
            ("chat", read_domain_list("chat")),
            ("browsing-shopping", read_domain_list(["forums", "blog", "press", "sports", "cooking", "cleaning", \
                                                    "shopping", "lingerie"])),
        ]
             
        
        def cat_search(x):
            SITES = ["yandex.ru", "yastatic.net", "baidu.com", "bdstatic.com", "duckduckgo.com", "ask.com", "bing.com", \
                     "google.com", "google.fr", "google.ca", \
                     "www.google.com", "www.google.fr", "www.google.ca"]
            SITES = set(SITES)
            if x in SITES:
                return len(x)
            else:
                return -1
        
        
        def f1(x):
            '''
            if re.match(RE_VIDEO2, x):
                return "video"
            '''    
            if re.match(RE_REMOVED, x):
                return "unknown"
            
            maxmatchs = [(name, longest_matching_domain(x, dom)) for name,dom in ENSEMBLE] \
                + [("search", cat_search(x))]
            name,count = max(maxmatchs, key=lambda x:x[1])

            if count == -1:
                return "unknown"
            else:
                return name
        def f2(x):
            y = f1(x)
            if y == 'unknown':
                return -1
            else:
                return NAMES.index(y)
        
        
        counts_map = {'social': 32715,
             'browsing-shopping': 33971+6801,
             'search': 9043,
             'mail': 12337,
             'chat': 5398,
             'download': 1151}
        counts = [counts_map[x] for x in NAMES]
        weights = counts_to_weights(list(enumerate(counts)))
        
        return Ensemble(NAMES, counts, weights, f2, cached=cached)

       
    @staticmethod
    def get_singleClass_ensemble(cached=True):

        NAMES = ["social"]

        ENSEMBLE = [
            ("social", read_domain_list("testAudioCDNs")),
           ]
        
        RE_REMOVED = r".*(scontent.*fbcdn\.net|static.*fbcdn\.net)"        
        def f1(x):
            
            if re.match(RE_REMOVED, x):
                return "unknown"

            maxmatchs = [(name, longest_matching_domain(x, dom)) for name,dom in ENSEMBLE]
            name,count = max(maxmatchs, key=lambda x:x[1])

            if count == -1:
                return "unknown"
            else:
                print(name)
                return name
        def f2(x):
            y = f1(x)
            if y == 'unknown':
                return -1
            else:
                return NAMES.index(y)

        counts_map = {'social': 1}
        counts = [counts_map[x] for x in NAMES]
        weights = counts_to_weights(list(enumerate(counts)))

        return Ensemble(NAMES, counts, weights, f2, cached=cached)
    

    @staticmethod
    def get_singleClassDiag_ensemble(cached=True):

        NAMES = ["diag"]

        ENSEMBLE = [
           ("diag",read_domain_list(["forums", "blog", "sports", "cooking", "cleaning", \
                                                   "shopping", "lingerie","press"]))]
                
        RE_REMOVED = r".*(scontent.*fbcdn\.net|static.*fbcdn\.net)"
        #Remove some fbcdn flows because they cant be labelled and might match on some other domains

        def f1(x):
           
            if re.match(RE_REMOVED, x):
                return "unknown"
        
            maxmatchs = [(name, longest_matching_domain_diag(x, dom)) for name,dom in ENSEMBLE]
            name,count = max(maxmatchs, key=lambda x:x[1])
            
            return count
   
        def f2(x):
            
            return f1(x)

        counts_map = {'diag': 1}
        counts = [counts_map[x] for x in NAMES]
        weights = counts_to_weights(list(enumerate(counts)))

        return Ensemble(NAMES, counts, weights, f2, cached=cached)
    
           
    @staticmethod
    def get_newClasses_ensemble(cached=True):
    
        NAMES = ["chat", "download", "games", "mail", "news", "search", "social", "streaming", "web"]
        
        ENSEMBLE = [
            ("streaming", read_domain_list("streaming")),
            ("chat", read_domain_list("chat")),
            ("download", read_domain_list("download")),
            ("social", read_domain_list("social_networks")),
            ("games", read_domain_list("games")),
            ("search", read_domain_list("search")),
            ("mail", read_domain_list("mail")),
            ("news", read_domain_list("press")),
            ("web", read_domain_list(["forums", "blog", "sports", "cooking", "cleaning", \
                                                   "shopping", "lingerie"]))]
        
        RE_REMOVED = r".*(scontent.*fbcdn\.net|static.*fbcdn\.net)"
        #Remove some fbcdn flows because they cant be labelled and might match on some other domains

            
        def f1(x):
            
            if re.match(RE_REMOVED, x):
                return "unknown"

            
            #domains = [re.compile(".*" + d) for name, dom in ENSEMBLE for d in dom]            
            #maxmatchs = [(name, longest_matching_domain_regex(x, domains)) for name,dom in ENSEMBLE]
            
            
            maxmatchs = [(name, longest_matching_domain(x, dom)) for name,dom in ENSEMBLE]
            name,count = max(maxmatchs, key=lambda x:x[1])

            if count == -1:
                return "unknown"
            else:
                return name
            
        def f2(x):
            y = f1(x)
            if y == 'unknown':
                return -1
            else:
                return NAMES.index(y)
        
        counts_map = {'chat': 1,
             'download': 1,
             'social' : 1,
             'games': 1,
             'mail': 1,
             'news': 1,
             'streaming': 1,
             'search': 1,
             'web':1,
             }

        counts = [counts_map[x] for x in NAMES]
        weights = counts_to_weights(list(enumerate(counts)))
        
        return Ensemble(NAMES, counts, weights, f2, cached=cached)
    
    @staticmethod
    def get_newClassesNoPress_ensemble(cached=True):
    
        NAMES = ["chat", "download", "games", "mail",  "search", "social", "streaming", "web"]
        
        ENSEMBLE = [
            ("streaming", read_domain_list("streaming")),
            ("chat", read_domain_list("chat")),
            ("download", read_domain_list("download")),
            ("social", read_domain_list("social_networks")),
            ("games", read_domain_list("games")),
            ("search", read_domain_list("search")),
            ("mail", read_domain_list("mail")),
            ("web", read_domain_list(["forums", "blog", "sports", "cooking", "cleaning", \
                                                   "shopping", "lingerie", "press"]))]
        
        RE_REMOVED = r".*(scontent.*fbcdn\.net|static.*fbcdn\.net)"
        #Remove some fbcdn flows because they cant be labelled and might match on some other domains

            
        def f1(x):
            
            if re.match(RE_REMOVED, x):
                return "unknown"
    
            
            maxmatchs = [(name, longest_matching_domain(x, dom)) for name,dom in ENSEMBLE]
            name,count = max(maxmatchs, key=lambda x:x[1])

            if count == -1:
                return "unknown"
            else:
                return name
            
        def f2(x):
            y = f1(x)
            if y == 'unknown':
                return -1
            else:
                return NAMES.index(y)
        
        counts_map = {'chat': 1,
             'download': 1,
             'social' : 1,
             'games': 1,
             'mail': 1,
             'streaming': 1,
             'search': 1,
             'web':1,
             }

        counts = [counts_map[x] for x in NAMES]
        weights = counts_to_weights(list(enumerate(counts)))
        
        return Ensemble(NAMES, counts, weights, f2, cached=cached)
    
    
## domain splitting tactics
def domain_split(domains, seed=3549, ratio=0.25, verbose=False):
    """
    Domains: list of (str, number)
    
    Return: Train domains, validation domains
    """
    random.shuffle(domains)
    total = sum([n for d,n in domains]) * ratio
    
    acc = 0
    p = 0
    for i in range(len(domains)):
        name,count = domains[i]
        if acc >= total * ratio:
            p = i
            break
        acc += count
        
    assert acc <= total
        
    if p == 0 or p == len(domains) - 1:
        sys.stderr.write("Warning: Domain splitting results in an empty list\n")
            
    if verbose:
        print("Cutoff point: {}, Real ratio: {}".format(p, acc  / total))
        
    domains_train = [x[0] for x in domains[p:]]
    domains_val = [x[0] for x in domains[:p]]
    return domains_train, domains_val

def domain_split_suffix(domains, seed=3549, ratio=0.25, verbose=False):
    groups = dict()
    for s,n in domains:
        suffix = domain_suffix(s)
        if suffix in groups:
            groups[suffix].append((s,n))
        else:
            groups[suffix] = [(s,n)]
            
    regroup = []
    for k,v in groups.items():
        doms = [x[0] for x in v]
        n = sum([x[1] for x in v])
        regroup.append((doms, n))
        
    domains_train, domains_val = domain_split(regroup, seed, ratio, verbose)
    
    domains_train = [x for y in domains_train for x in y]
    domains_val = [x for y in domains_val for x in y]
    
    return domains_train, domains_val