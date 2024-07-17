import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule

class MySpider(CrawlSpider):
    name = 'spider_d5'
    allowed_domains = ['docs.nvidia.com']
    start_urls = ['https://docs.nvidia.com/cuda/']  

    rules = (
        Rule(LinkExtractor(), callback='parse_item', follow=True),
    )

    custom_settings = {
        'DEPTH_LIMIT': 4, 
        'DEPTH_PRIORITY': 1,  
    }

    def parse_item(self, response):
        page_text = ' '.join(response.xpath('//body//text()').extract()).strip()
        yield {
            'url': response.url,
            'title': response.css('title::text').get(),
            'content': page_text
        }