---
title: "프로젝트"
layout: archive
permalink: /projects/
author_profile: false
classes: wide
---

<div class="page-intro">
  <p class="section-kicker">Projects</p>
  <p>문제 정의부터 데이터 해석, 실험 비교, 결과와 한계까지 실제 작업 흐름을 기준으로 정리합니다.</p>
</div>

<div class="project-grid project-grid--page">
  {% for project in site.data.projects %}
    <article class="project-card">
      <span class="project-card__label">{{ project.label }}</span>
      <h2>{% if project.url %}<a href="{{ project.url | relative_url }}">{{ project.title }}</a>{% else %}{{ project.title }}{% endif %}</h2>
      <p>{{ project.description }}</p>
    </article>
  {% endfor %}
</div>
