#!/usr/bin/env python3
"""
查看OpenAlex数据的脚本
支持多种查询方式：关键词搜索、作者、机构、期刊等
"""
import json
import requests
import argparse
from typing import List, Dict, Any
from datetime import datetime


class OpenAlexViewer:
    """OpenAlex数据查看器"""
    
    BASE_URL = "https://api.openalex.org"
    
    def __init__(self, email: str = None):
        """
        初始化
        
        Args:
            email: 可选的邮箱地址，用于提高API速率限制
        """
        self.email = email
        self.session = requests.Session()
        if email:
            self.session.headers.update({"User-Agent": f"mailto:{email}"})
    
    def search_works(
        self,
        query: str = None,
        title: str = None,
        author: str = None,
        venue: str = None,
        year: int = None,
        per_page: int = 25,
        page: int = 1,
        sort: str = "cited_by_count:desc"
    ) -> Dict[str, Any]:
        """
        搜索论文（works）
        
        Args:
            query: 关键词搜索
            title: 标题搜索
            author: 作者搜索
            venue: 期刊/会议搜索
            year: 年份
            per_page: 每页结果数
            page: 页码
            sort: 排序方式
            
        Returns:
            API响应结果
        """
        params = {
            "per-page": per_page,
            "page": page,
            "sort": sort
        }
        
        # 构建搜索查询
        search_parts = []
        if query:
            search_parts.append(query)
        if title:
            search_parts.append(f"title.search:{title}")
        if author:
            search_parts.append(f"author.search:{author}")
        if venue:
            search_parts.append(f"venue.search:{venue}")
        if year:
            search_parts.append(f"publication_year:{year}")
        
        if search_parts:
            params["search"] = " | ".join(search_parts)
        
        try:
            response = self.session.get(f"{self.BASE_URL}/works", params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"搜索错误: {e}")
            return {}
    
    def get_work_by_id(self, work_id: str) -> Dict[str, Any]:
        """
        根据ID获取论文详情
        
        Args:
            work_id: OpenAlex工作ID（可以是W123456789格式或URL）
            
        Returns:
            论文详情
        """
        # 处理不同的ID格式
        if work_id.startswith("W"):
            work_id = work_id[1:]
        if work_id.startswith("https://openalex.org/"):
            work_id = work_id.split("/")[-1]
        
        try:
            response = self.session.get(f"{self.BASE_URL}/works/W{work_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"获取论文详情错误: {e}")
            return {}
    
    def format_work(self, work: Dict[str, Any], detailed: bool = False) -> str:
        """
        格式化论文信息
        
        Args:
            work: 论文数据
            detailed: 是否显示详细信息
            
        Returns:
            格式化后的字符串
        """
        if not work:
            return "无数据"
        
        lines = []
        
        # 基本信息
        title = work.get("title", "N/A")
        lines.append(f"标题: {title}")
        
        if detailed:
            # 显示ID
            work_id = work.get("id", "").split("/")[-1] if work.get("id") else "N/A"
            lines.append(f"ID: {work_id}")
            
            # 显示DOI
            doi = work.get("doi", "N/A")
            if doi:
                lines.append(f"DOI: {doi}")
            
            # 显示发表年份
            year = work.get("publication_year", "N/A")
            lines.append(f"年份: {year}")
            
            # 显示期刊/会议
            venue = work.get("primary_location", {}).get("venue", {})
            if venue:
                venue_name = venue.get("display_name", "N/A")
                lines.append(f"期刊/会议: {venue_name}")
            
            # 显示作者
            authorships = work.get("authorships", [])
            if authorships:
                author_names = [a.get("author", {}).get("display_name", "N/A") 
                              for a in authorships[:5]]
                lines.append(f"作者: {', '.join(author_names)}")
                if len(authorships) > 5:
                    lines.append(f"  ... 共{len(authorships)}位作者")
            
            # 显示摘要
            abstract = work.get("abstract", "")
            if abstract:
                abstract_text = abstract.get("abstract", "") if isinstance(abstract, dict) else abstract
                if abstract_text:
                    lines.append(f"摘要: {abstract_text[:200]}...")
            
            # 显示引用数
            cited_by_count = work.get("cited_by_count", 0)
            lines.append(f"引用数: {cited_by_count}")
            
            # 显示概念（主题）
            concepts = work.get("concepts", [])
            if concepts:
                concept_names = [c.get("display_name", "") for c in concepts[:5]]
                lines.append(f"主题: {', '.join(concept_names)}")
            
            # 显示URL
            open_access = work.get("open_access", {})
            if open_access.get("is_oa"):
                oa_url = open_access.get("oa_url", "")
                if oa_url:
                    lines.append(f"开放获取URL: {oa_url}")
            
            primary_location = work.get("primary_location", {})
            landing_page_url = primary_location.get("landing_page_url", "")
            if landing_page_url:
                lines.append(f"访问链接: {landing_page_url}")
        else:
            # 简化信息
            year = work.get("publication_year", "")
            cited_by_count = work.get("cited_by_count", 0)
            lines.append(f"年份: {year}, 引用数: {cited_by_count}")
        
        return "\n".join(lines)
    
    def list_works(
        self,
        query: str = None,
        title: str = None,
        author: str = None,
        venue: str = None,
        year: int = None,
        limit: int = 25,
        detailed: bool = False,
        output_file: str = None
    ):
        """
        列出论文并显示
        
        Args:
            query: 关键词搜索
            title: 标题搜索
            author: 作者搜索
            venue: 期刊/会议搜索
            year: 年份
            limit: 显示数量限制
            detailed: 是否显示详细信息
            output_file: 输出文件路径（JSON格式）
        """
        print("正在搜索OpenAlex数据...")
        print("-" * 80)
        
        results = []
        per_page = min(limit, 200)  # OpenAlex每页最多200条
        total_pages = (limit + per_page - 1) // per_page
        
        for page in range(1, total_pages + 1):
            response = self.search_works(
                query=query,
                title=title,
                author=author,
                venue=venue,
                year=year,
                per_page=per_page,
                page=page
            )
            
            works = response.get("results", [])
            if not works:
                break
            
            results.extend(works)
            
            if len(results) >= limit:
                results = results[:limit]
                break
        
        # 显示结果
        print(f"\n找到 {len(results)} 条结果\n")
        print("=" * 80)
        
        for i, work in enumerate(results, 1):
            print(f"\n[{i}/{len(results)}]")
            print(self.format_work(work, detailed=detailed))
            print("-" * 80)
        
        # 保存到文件
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n数据已保存到: {output_file}")
        
        # 显示字段统计
        if results:
            print("\n" + "=" * 80)
            print("数据字段统计:")
            print(f"  总记录数: {len(results)}")
            
            # 统计有各种字段的记录数
            has_doi = sum(1 for w in results if w.get("doi"))
            has_abstract = sum(1 for w in results if w.get("abstract"))
            has_citations = sum(1 for w in results if w.get("cited_by_count", 0) > 0)
            has_open_access = sum(1 for w in results if w.get("open_access", {}).get("is_oa", False))
            
            print(f"  有DOI: {has_doi} ({has_doi/len(results)*100:.1f}%)")
            print(f"  有摘要: {has_abstract} ({has_abstract/len(results)*100:.1f}%)")
            print(f"  有引用: {has_citations} ({has_citations/len(results)*100:.1f}%)")
            print(f"  开放获取: {has_open_access} ({has_open_access/len(results)*100:.1f}%)")
            
            # 显示示例字段
            if results:
                print("\n示例字段结构:")
                example = results[0]
                print(f"  主要字段: {', '.join(example.keys())}")


def main():
    parser = argparse.ArgumentParser(
        description="查看OpenAlex数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 关键词搜索
  python view_openalex_data.py --query "machine learning"
  
  # 按作者搜索
  python view_openalex_data.py --author "Yann LeCun"
  
  # 按期刊搜索
  python view_openalex_data.py --venue "Nature"
  
  # 按年份和关键词搜索
  python view_openalex_data.py --query "transformer" --year 2023
  
  # 显示详细信息并保存到文件
  python view_openalex_data.py --query "LLM" --limit 10 --detailed --output results.json
        """
    )
    
    parser.add_argument("--query", "-q", type=str, help="关键词搜索")
    parser.add_argument("--title", "-t", type=str, help="标题搜索")
    parser.add_argument("--author", "-a", type=str, help="作者搜索")
    parser.add_argument("--venue", "-v", type=str, help="期刊/会议搜索")
    parser.add_argument("--year", "-y", type=int, help="年份")
    parser.add_argument("--limit", "-n", type=int, default=25, help="显示数量限制（默认25）")
    parser.add_argument("--detailed", "-d", action="store_true", help="显示详细信息")
    parser.add_argument("--output", "-o", type=str, help="输出JSON文件路径")
    parser.add_argument("--email", "-e", type=str, help="邮箱地址（用于提高API速率限制）")
    
    args = parser.parse_args()
    
    # 检查是否有任何搜索条件
    if not any([args.query, args.title, args.author, args.venue, args.year]):
        parser.print_help()
        print("\n错误: 请至少提供一个搜索条件（--query, --title, --author, --venue, --year）")
        return
    
    viewer = OpenAlexViewer(email=args.email)
    viewer.list_works(
        query=args.query,
        title=args.title,
        author=args.author,
        venue=args.venue,
        year=args.year,
        limit=args.limit,
        detailed=args.detailed,
        output_file=args.output
    )


if __name__ == "__main__":
    main()
