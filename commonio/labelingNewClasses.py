import re
import random
import sys
import os
from collections import Counter

RE_VIDEO = r".*(xx\.fbcdn|xhcdn|t8cdn|phncdn|xvideos-cdn|xhcdn.com|xnxx-cdn|ypncdn|t8cdn|sb-cd.com|dailymotion|googlevideo\.com|rdtcdn|nflxvideo|vod.*akamaized|ttvnw.net|vid.*cdn|video.twimg|vod.*cdn|bcrncdn|dditscdn|streaming.estat|cdn.*vid|wlmediahub)"


def is_video(x):
    return re.match(RE_VIDEO, x)

def read_domain_list_legacy(name):
    #Legacy function used to create ensembles based on the old set of classes
    if not isinstance(name, list):
        name = [name]
    lines = []
    for n in name:
        with open("/home/nmalekgh/deep_traffic_git/orange-deep-traffic/assets/domains_thethys/ut_capitole/{}/domains".format(n)) as f:
            lines += [line.rstrip() for line in f]
    lines = sorted(lines, key=lambda x:-len(x))
    return lines


####This function is not backwards compatible, please use read_domain_list_legacy to create ensemble on the old set of classes
def read_domain_list(name):
    ##This will open the new homemde dataset in priority and then a modified version of the UT1 dataset
    
    if not isinstance(name, list):
        name = [name]
    lines = []
    for n in name:
        #This functions will open the homemade dataset in priority
        #If no homemade dataset with the required name is found, it will instead open the corresponding UT1 file
        if(os.path.isfile("/home/nmalekgh/deep_traffic_git/orange-deep-traffic/assets/domains_thethys/homemade_dataset/domainsTest/{}".format(n))):
            with open("/home/nmalekgh/deep_traffic_git/orange-deep-traffic/assets/domains_thethys/homemade_dataset/domainsTest/{}".format(n)) as f:
                lines += [line.rstrip() for line in f]
        else:
            #This path is for the modified ut1 dataset.
            with open("/home/nmalekgh/deep_traffic_git/orange-deep-traffic/assets/domains_thethys/homemade_dataset/ut_capitole_modified/{}/domains".format(n)) as f:
                lines += [line.rstrip() for line in f]

    lines = sorted(lines, key=lambda x:-len(x))
    
    return lines

def is_in_domain(addr, domains):
    return any(map(lambda x:addr.endswith(x), domains))
def is_in_domain2(addr, domains):
    return addr in domains

def longest_matching_domain(addr, domains):
    #This function was changed to handle regex in the domains written in files.
    #It is backwards compatible as the UT1 dataset does not use regex.
    
    for d in domains:
        #Quick&dirty trick to reduce the complexity. Use the regex compiler only when needed
        #Also ensures backwards compatibility
        if("*" in d or "^" in d):
            match = re.match(".*"+d,addr)
            lenMatch = len(match.group(0)) if not(match is None) else 0  
            if (lenMatch > 0):
                return lenMatch
    
        elif (addr.endswith(d)):
            return len(d)     
            
    return -1

################################# DIAGNOSTIC FUNCTION DO NOT USE TO TRAIN #################################
def longest_matching_domain_diag(addr, domains):
    #Diag function used to dump the matched domains from a single class to a file.
    #Using this function instead of the regular one causes the labels variable to contain a 
    #string with the matched domain instead of the index of the matched class
    #This function crashes the rest of the preporcessing but allows for troubleshooting and noise detection
    
    for d in domains:
        
        if("*" in d or "^" in d) :
            if(re.match(".*"+d,addr)):
                return("Regex matched addr: "+ addr + " to domain: " +d)
            
        else :
            if(addr.endswith(d) and (not addr==d)):
                return("Matched addr: " + addr + " to domain: " + d)

            if addr.endswith(d):

                return("Exact match found on: " + d) 

    return -1


LABELS = ['video', 'browsing', 'social', 'download', '']

def counts_to_weights(counts, weight_cap=None):
    """
    Input: [(class, count)]
    Output: { class: weight }
    """
    
    max_count = max([c for x,c in counts])
    result = dict()
    for x, c in counts:
        assert c >= 0
        if c == 0:
            print("WARNING: Class {} has count {} == 0".format(x, c))
            c = 1e-10
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
        #self.counts = [counts_map[i] for i in range(len(self.names))]
        self.counts = [counts_map.get(i, 0) for i in range(len(self.names))]
        self.weights = counts_to_weights(list(enumerate(self.counts)), weight_cap)
        
    

    @staticmethod
    def get_0214_ensemble(cached=True):
        """
        The ensemble is generated from orange TLS dataset. It is used in the multi-class model on Feb. 14.
        """
        domains_socialmedia = read_domain_list_legacy("social_networks")
        domains_mail        = read_domain_list_legacy(["mail", "webmail"])
        domains_chat        = read_domain_list_legacy("chat")
        domains_browsing    = read_domain_list_legacy(["forums", "blog", "press", "sports", "cooking", "cleaning"])
        domains_shopping    = read_domain_list_legacy(["shopping", "lingerie"])
        domains_ads         = read_domain_list_legacy("ads")
        domains_redir       = read_domain_list_legacy(["shortener", "redirector", "strong_redirector", "strict_redirector"])

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
            ("social", read_domain_list_legacy("social_networks")),
            ("mail", read_domain_list_legacy(["mail", "webmail"])),
            ("download", read_domain_list_legacy(["update", "download", "filehosting"])),
            ("chat", read_domain_list_legacy("chat")),
            ("browsing", read_domain_list_legacy(["forums", "blog", "press", "sports", "cooking", "cleaning"])),
            ("shopping", read_domain_list_legacy(["shopping", "lingerie"])),
            ("ads", read_domain_list_legacy("ads")),
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
            ("social", read_domain_list_legacy("social_networks")),
            ("mail", read_domain_list_legacy(["mail", "webmail"])),
            ("download", read_domain_list_legacy(["update", "download", "filehosting"])),
            ("chat", read_domain_list_legacy("chat")),
            ("browsing", read_domain_list_legacy(["forums", "blog", "press", "sports", "cooking", "cleaning"])),
            ("shopping", read_domain_list_legacy(["shopping", "lingerie"])),
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
            ("social", read_domain_list_legacy("social_networks")),
            ("mail", read_domain_list_legacy(["mail", "webmail"])),
            ("download", read_domain_list_legacy(["update", "download", "filehosting"])),
            ("chat", read_domain_list_legacy("chat")),
            ("browsing-shopping", read_domain_list_legacy(["forums", "blog", "press", "sports", "cooking", "cleaning", \
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

        
    ################################# DIAGNOSTIC FUNCTION DO NOT USE TO TRAIN #################################
    @staticmethod
    def get_singleClassDiag_ensemble(cached=True):
        #This function is used to dump the matches of a single class.
        #When running the pre processing with this ensemble, the array containing the index of classes will instead
        #contain the string indicating what SNI was matched on which domain

        NAMES = ["diag"]

        ENSEMBLE = [
            #Put in the domain list whatever lookup table you want to match against
           ("diag", read_domain_list("streaming"))]
                
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
        #Creates the ensemble for the new classes
    
        #The streaming and search classes have been moved to the server to and are no longer hardcoded
        NAMES = ["chat", "download", "games", "mail", "news", "search", "social", "streaming", "web"]
        
        
        #DISCLAIMER : the web class stil has interaction with other classes and should remain last in this list.
        # Changing its position in the declaration of ENSEMBLE may impact the resulting labeling
        # Streaming also has some (very) minor interactions with social regrding some specific fbcdn flows
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
        #Creates the ensemble for the new classes but press was merged into web
        #These classes are refered to as Final classes in the model experiments
    
    
        #The streaming and search classes have been moved to the server to and are no longer hardcoded
        NAMES = ["chat", "download", "games", "mail",  "search", "social", "streaming", "web"]
        
        #DISCLAIMER : the web class stil has interaction with other classes and should remain last in this list.
        # Changing its position in the declaration of games may impact the resulting ensemble
        # Streaming also has some (very) minor interactions with social regrding some specific fbcdn flows
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
    
    @staticmethod    
    def get_newClassesNoPress_ensemble_NoWebClass(cached=True):
        #same as above function without 
    
    
        #The streaming and search classes have been moved to the server to and are no longer hardcoded
        NAMES = ["chat", "download", "games", "mail",  "search", "social", "streaming"]
        
        #DISCLAIMER : the web class stil has interaction with other classes and should remain last in this list.
        # Changing its position in the declaration of games may impact the resulting ensemble
        # Streaming also has some (very) minor interactions with social regrding some specific fbcdn flows
        ENSEMBLE = [
            ("streaming", read_domain_list("streaming")),
            ("chat", read_domain_list("chat")),
            ("download", read_domain_list("download")),
            ("social", read_domain_list("social_networks")),
            ("games", read_domain_list("games")),
            ("search", read_domain_list("search")),
            ("mail", read_domain_list("mail"))]
        
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
             'search': 1
             }

        counts = [counts_map[x] for x in NAMES]
        weights = counts_to_weights(list(enumerate(counts)))
        
        return Ensemble(NAMES, counts, weights, f2, cached=cached)
    
    
    @staticmethod    
    def get_QUICpreliminary_newClassesOnlySocialStreamingWeb_ensemble(cached=True):
        #same as above function without 
    
    
        #The streaming and search classes have been moved to the server to and are no longer hardcoded
        NAMES = ["social", "streaming", "web"]
        
        #DISCLAIMER : the web class stil has interaction with other classes and should remain last in this list.
        # Changing its position in the declaration of games may impact the resulting ensemble
        # Streaming also has some (very) minor interactions with social regrding some specific fbcdn flows
        ENSEMBLE = [
            ("streaming", read_domain_list("streaming")),
            ("social", read_domain_list("social_networks")),
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
        
        counts_map = {'social': 1,
             'streaming': 1,
             'web': 1
             }

        counts = [counts_map[x] for x in NAMES]
        weights = counts_to_weights(list(enumerate(counts)))
        
        return Ensemble(NAMES, counts, weights, f2, cached=cached)
    
    @staticmethod    
    def get_QUICpreliminary_newClassesOnlySocialWeb_ensemble(cached=True):
        #same as above function without 
    
    
        #The streaming and search classes have been moved to the server to and are no longer hardcoded
        NAMES = ["social", "web"]
        
        #DISCLAIMER : the web class stil has interaction with other classes and should remain last in this list.
        # Changing its position in the declaration of games may impact the resulting ensemble
        # Streaming also has some (very) minor interactions with social regrding some specific fbcdn flows
        ENSEMBLE = [
            ("social", read_domain_list("social_networks")),
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
        
        counts_map = {'social': 1,
             'streaming': 1,
             'web': 1
             }

        counts = [counts_map[x] for x in NAMES]
        weights = counts_to_weights(list(enumerate(counts)))
        
        return Ensemble(NAMES, counts, weights, f2, cached=cached)
    
    
    @staticmethod
    def get_ApplicationClasses_ensemble(cached=True):
        #This methods creates an ensemble for application level classificatio.
        #Please read the README file in the "/home/orange/domains/homemade_dataset/domainsTest/" folder for more details
    
    
        NAMES = ["chatFacebook", "chatSnapchat", "chatWhatsapp", "downloadApple", "downloadGooglePlay", "mailGmail",\
                 "mailHotmail", "mailOutlook", "searchGoogle", "socialFacebook", "socialInstagram", "socialTwitter",\
                 "streamingFacebook", "streamingNetflix", "streamingSnapchat", "streamingYoutube", "webAmazon",\
                 "webAppleLocalization", "webMicrosoft"]

        ENSEMBLE = [
            ("chatFacebook", read_domain_list("chatFacebook")),
            ("chatSnapchat", read_domain_list("chatSnapchat")),
            ("chatWhatsapp", read_domain_list("chatWhatsapp")),
            ("downloadApple", read_domain_list("downloadApple")),
            ("downloadGooglePlay", read_domain_list("downloadGooglePlay")),
            ("mailGmail", read_domain_list("mailGmail")),
            ("mailHotmail", read_domain_list("mailHotmail")),
            ("mailOutlook", read_domain_list("mailOutlook")),
            ("searchGoogle", read_domain_list("searchGoogle")),
            ("socialFacebook", read_domain_list("socialFacebook")),
            ("socialInstagram", read_domain_list("socialInstagram")),
            ("socialTwitter", read_domain_list("socialTwitter")),
            ("streamingFacebook", read_domain_list("streamingFacebook")),
            ("streamingNetflix", read_domain_list("streamingNetflix")),
            ("streamingSnapchat", read_domain_list("streamingSnapchat")),
            ("streamingYoutube", read_domain_list("streamingYoutube")),
            ("webAmazon", read_domain_list("webAmazon")),
            ("webAppleLocalization", read_domain_list("webAppleLocalization")),
            ("webMicrosoft", read_domain_list("webMicrosoft")),
            ]
        
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
        
        counts_map = {"chatFacebook": 1,
             "chatSnapchat": 1,
             "chatWhatsapp":1,
             "downloadApple":1,
             "downloadGooglePlay":1,
             "mailGmail":1,
             "mailHotmail":1,
             "mailOutlook":1, 
             "searchGoogle":1,
             "socialFacebook":1,
             "socialInstagram":1,
             "socialTwitter":1,
             "streamingFacebook":1,
             "streamingNetflix":1,
             "streamingSnapchat":1,
             "streamingYoutube":1,
             "webAmazon":1,
             "webAppleLocalization":1,
             "webMicrosoft":1
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