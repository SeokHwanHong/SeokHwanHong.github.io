---
title: "프로젝트"
layout: archive
permalink: /projects/
author_profile: false
classes: wide
---

<div class="page-intro">
  <p class="section-kicker">Projects</p>
  <p>데이터를 이해하고 모델과 서비스로 연결하는 과정에서 수행한 프로젝트를 정리합니다.</p>
</div>

<div class="project-grid project-grid--page">
  {% for project in site.data.projects %}
    <article class="project-card">
      <span class="project-card__label">{{ project.label }}</span>
      <h2>{{ project.title }}</h2>
      <p>{{ project.description }}</p>
    </article>
  {% endfor %}
</div>
