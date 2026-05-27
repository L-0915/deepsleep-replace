import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, ChevronRight, ChevronLeft, CheckCircle } from 'lucide-react';
import useChatStore from '../hooks/useChat';

const questions = [
  {
    id: 'bedtime',
    text: '过去一个月，你通常几点上床睡觉？',
    type: 'time',
  },
  {
    id: 'fallAsleep',
    text: '通常多久能入睡？',
    type: 'slider',
    min: 0,
    max: 60,
    step: 5,
    unit: '分钟',
  },
  {
    id: 'sleepHours',
    text: '每天实际睡多少小时？',
    type: 'slider',
    min: 3,
    max: 12,
    step: 0.5,
    unit: '小时',
  },
  {
    id: 'wakeUp',
    text: '夜间醒来几次？',
    type: 'select',
    options: [
      { value: '0', label: '0 次' },
      { value: '1-2', label: '1-2 次' },
      { value: '3-4', label: '3-4 次' },
      { value: '5+', label: '5 次以上' },
    ],
  },
  {
    id: 'quality',
    text: '睡眠质量自我评价',
    type: 'rating',
    labels: ['很差', '较差', '一般', '较好', '很好'],
  },
  {
    id: 'daytimeEnergy',
    text: '白天精力如何？',
    type: 'rating',
    labels: ['非常差', '较差', '一般', '较好', '非常好'],
  },
  {
    id: 'snoring',
    text: '是否打鼾？',
    type: 'select',
    options: [
      { value: 'no', label: '否' },
      { value: 'yes', label: '是' },
      { value: 'unsure', label: '不确定' },
    ],
  },
];

function calculateScore(answers) {
  let score = 0;

  // Fall asleep time (0-5min=0, 5-15=1, 15-30=2, 30-60=3, >60=4)
  const fallAsleep = parseFloat(answers.fallAsleep || 0);
  if (fallAsleep <= 5) score += 0;
  else if (fallAsleep <= 15) score += 1;
  else if (fallAsleep <= 30) score += 2;
  else if (fallAsleep <= 60) score += 3;
  else score += 4;

  // Sleep hours (>=8=0, 7-8=1, 6-7=2, 5-6=3, <5=4)
  const hours = parseFloat(answers.sleepHours || 7);
  if (hours >= 8) score += 0;
  else if (hours >= 7) score += 1;
  else if (hours >= 6) score += 2;
  else if (hours >= 5) score += 3;
  else score += 4;

  // Wake up count
  const wakeUp = answers.wakeUp || '0';
  if (wakeUp === '0') score += 0;
  else if (wakeUp === '1-2') score += 1;
  else if (wakeUp === '3-4') score += 2;
  else score += 3;

  // Quality
  const quality = parseInt(answers.quality || 3);
  score += (5 - quality);

  // Daytime energy
  const energy = parseInt(answers.daytimeEnergy || 3);
  score += (5 - energy);

  // Snoring
  if (answers.snoring === 'yes') score += 2;
  else if (answers.snoring === 'unsure') score += 1;

  return Math.min(score, 21);
}

function getScoreLevel(score) {
  if (score <= 4) return { level: '优秀', color: '#22c55e', desc: '你的睡眠质量非常好，继续保持良好的睡眠习惯。' };
  if (score <= 8) return { level: '良好', color: '#84cc16', desc: '你的睡眠质量总体良好，仍有提升空间。' };
  if (score <= 12) return { level: '一般', color: '#eab308', desc: '你的睡眠质量有待改善，建议关注睡眠卫生。' };
  if (score <= 16) return { level: '较差', color: '#f97316', desc: '你的睡眠质量较差，建议采取措施改善睡眠。' };
  return { level: '很差', color: '#ef4444', desc: '你的睡眠质量很差，强烈建议咨询专业医生。' };
}

export default function SleepAssessment() {
  const { assessmentOpen, setAssessmentOpen } = useChatStore();
  const [step, setStep] = useState(0);
  const [answers, setAnswers] = useState({});
  const [showResult, setShowResult] = useState(false);

  const currentQuestion = questions[step];
  const isLastStep = step === questions.length - 1;
  const score = calculateScore(answers);
  const scoreInfo = getScoreLevel(score);

  const setAnswer = (id, value) => {
    setAnswers(prev => ({ ...prev, [id]: value }));
  };

  const handleNext = () => {
    if (isLastStep) {
      setShowResult(true);
    } else {
      setStep(step + 1);
    }
  };

  const handlePrev = () => {
    if (showResult) {
      setShowResult(false);
    } else if (step > 0) {
      setStep(step - 1);
    }
  };

  const handleClose = () => {
    setAssessmentOpen(false);
    setStep(0);
    setAnswers({});
    setShowResult(false);
  };

  const handleSendToAI = () => {
    const resultText = `我的睡眠评估结果：\n` +
      `- 上床时间: ${answers.bedtime || '未填'}\n` +
      `- 入睡时长: ${answers.fallAsleep || '未填'} 分钟\n` +
      `- 实际睡眠: ${answers.sleepHours || '未填'} 小时\n` +
      `- 夜间醒来: ${answers.wakeUp || '未填'} 次\n` +
      `- 睡眠质量自评: ${answers.quality || '未填'}/5\n` +
      `- 白天精力: ${answers.daytimeEnergy || '未填'}/5\n` +
      `- 是否打鼾: ${answers.snoring === 'yes' ? '是' : answers.snoring === 'no' ? '否' : '不确定'}\n` +
      `- 综合评分: ${score}/21 (${scoreInfo.level})\n\n` +
      `请帮我分析我的睡眠状况，并给出改善建议。`;

    // Use a custom event to send to chat
    window.dispatchEvent(new CustomEvent('deepsleep-send-message', { detail: resultText }));
    handleClose();
  };

  const renderQuestion = () => {
    if (!currentQuestion) return null;

    switch (currentQuestion.type) {
      case 'time':
        return (
          <input
            type="time"
            value={answers[currentQuestion.id] || '23:00'}
            onChange={e => setAnswer(currentQuestion.id, e.target.value)}
            style={{
              width: '100%',
              padding: '12px 14px',
              background: 'var(--bg-card)',
              border: '1px solid var(--border)',
              borderRadius: 10,
              color: 'var(--text-primary)',
              fontSize: 16,
              fontFamily: 'inherit',
            }}
          />
        );

      case 'slider':
        return (
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
              <span style={{ color: 'var(--text-secondary)', fontSize: 13 }}>
                {currentQuestion.min} {currentQuestion.unit}
              </span>
              <span style={{ color: 'var(--accent)', fontSize: 16, fontWeight: 600 }}>
                {answers[currentQuestion.id] || currentQuestion.min} {currentQuestion.unit}
              </span>
              <span style={{ color: 'var(--text-secondary)', fontSize: 13 }}>
                {currentQuestion.max} {currentQuestion.unit}
              </span>
            </div>
            <input
              type="range"
              min={currentQuestion.min}
              max={currentQuestion.max}
              step={currentQuestion.step}
              value={answers[currentQuestion.id] || currentQuestion.min}
              onChange={e => setAnswer(currentQuestion.id, e.target.value)}
              style={{ width: '100%', accentColor: 'var(--accent)', cursor: 'pointer' }}
            />
          </div>
        );

      case 'select':
        return (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {currentQuestion.options.map(opt => (
              <button
                key={opt.value}
                onClick={() => setAnswer(currentQuestion.id, opt.value)}
                style={{
                  padding: '12px 16px',
                  background: answers[currentQuestion.id] === opt.value ? 'var(--accent)' : 'var(--bg-card)',
                  border: `1px solid ${answers[currentQuestion.id] === opt.value ? 'var(--accent)' : 'var(--border)'}`,
                  borderRadius: 10,
                  color: answers[currentQuestion.id] === opt.value ? '#fff' : 'var(--text-primary)',
                  cursor: 'pointer',
                  fontSize: 14,
                  textAlign: 'left',
                  transition: 'all 0.15s',
                }}
              >
                {opt.label}
              </button>
            ))}
          </div>
        );

      case 'rating':
        return (
          <div style={{ display: 'flex', gap: 8 }}>
            {currentQuestion.labels.map((label, i) => (
              <button
                key={i}
                onClick={() => setAnswer(currentQuestion.id, String(i + 1))}
                style={{
                  flex: 1,
                  padding: '12px 8px',
                  background: parseInt(answers[currentQuestion.id]) === i + 1 ? 'var(--accent)' : 'var(--bg-card)',
                  border: `1px solid ${parseInt(answers[currentQuestion.id]) === i + 1 ? 'var(--accent)' : 'var(--border)'}`,
                  borderRadius: 10,
                  color: parseInt(answers[currentQuestion.id]) === i + 1 ? '#fff' : 'var(--text-primary)',
                  cursor: 'pointer',
                  fontSize: 13,
                  textAlign: 'center',
                  transition: 'all 0.15s',
                }}
              >
                <div style={{ fontWeight: 600, marginBottom: 2 }}>{i + 1}</div>
                <div style={{ fontSize: 11, opacity: 0.8 }}>{label}</div>
              </button>
            ))}
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <AnimatePresence>
      {assessmentOpen && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={handleClose}
            style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.5)', zIndex: 300 }}
          />
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            transition={{ duration: 0.2 }}
            style={{
              position: 'fixed',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)',
              width: 500,
              maxWidth: '90vw',
              maxHeight: '90vh',
              background: 'var(--bg-secondary)',
              borderRadius: 16,
              border: '1px solid var(--border)',
              zIndex: 400,
              display: 'flex',
              flexDirection: 'column',
              overflow: 'hidden',
            }}
          >
            {/* Header */}
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '16px 20px',
                borderBottom: '1px solid var(--border)',
              }}
            >
              <h2 style={{ fontSize: 18, fontWeight: 600 }}>睡眠质量评估</h2>
              <button
                onClick={handleClose}
                style={{
                  background: 'none',
                  border: 'none',
                  color: 'var(--text-primary)',
                  cursor: 'pointer',
                  padding: 4,
                  borderRadius: 6,
                  display: 'flex',
                }}
              >
                <X size={20} />
              </button>
            </div>

            {/* Progress bar */}
            <div style={{ padding: '0 20px', paddingTop: 12 }}>
              <div
                style={{
                  height: 4,
                  background: 'var(--border)',
                  borderRadius: 2,
                  overflow: 'hidden',
                }}
              >
                <div
                  style={{
                    height: '100%',
                    background: 'var(--accent)',
                    borderRadius: 2,
                    transition: 'width 0.3s ease',
                    width: showResult ? '100%' : `${((step + 1) / questions.length) * 100}%`,
                  }}
                />
              </div>
              <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 4 }}>
                {showResult ? '评估完成' : `第 ${step + 1} 题 / 共 ${questions.length} 题`}
              </div>
            </div>

            {/* Content */}
            <div style={{ flex: 1, overflowY: 'auto', padding: '20px' }}>
              {showResult ? (
                <div style={{ textAlign: 'center' }}>
                  <CheckCircle size={48} style={{ color: scoreInfo.color, marginBottom: 12 }} />
                  <h3 style={{ fontSize: 22, fontWeight: 700, marginBottom: 4 }}>
                    睡眠评分: {score}/21
                  </h3>
                  <div
                    style={{
                      display: 'inline-block',
                      padding: '4px 16px',
                      borderRadius: 20,
                      background: `${scoreInfo.color}20`,
                      color: scoreInfo.color,
                      fontSize: 14,
                      fontWeight: 600,
                      marginBottom: 16,
                    }}
                  >
                    {scoreInfo.level}
                  </div>
                  <p style={{ color: 'var(--text-secondary)', fontSize: 14, lineHeight: 1.6 }}>
                    {scoreInfo.desc}
                  </p>

                  {/* Score breakdown */}
                  <div
                    style={{
                      marginTop: 20,
                      textAlign: 'left',
                      background: 'var(--bg-card)',
                      borderRadius: 10,
                      padding: 14,
                      border: '1px solid var(--border)',
                    }}
                  >
                    <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 8 }}>评估详情</div>
                    {[
                      { label: '上床时间', value: answers.bedtime || '未填' },
                      { label: '入睡时长', value: `${answers.fallAsleep || 0} 分钟` },
                      { label: '实际睡眠', value: `${answers.sleepHours || 0} 小时` },
                      { label: '夜间醒来', value: answers.wakeUp || '0 次' },
                      { label: '质量自评', value: `${answers.quality || '-'}/5` },
                      { label: '白天精力', value: `${answers.daytimeEnergy || '-'}/5` },
                      { label: '是否打鼾', value: answers.snoring === 'yes' ? '是' : answers.snoring === 'no' ? '否' : '不确定' },
                    ].map((item, i) => (
                      <div
                        key={i}
                        style={{
                          display: 'flex',
                          justifyContent: 'space-between',
                          padding: '6px 0',
                          borderBottom: i < 6 ? '1px solid var(--border)' : 'none',
                          fontSize: 13,
                        }}
                      >
                        <span style={{ color: 'var(--text-secondary)' }}>{item.label}</span>
                        <span>{item.value}</span>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div>
                  <h3 style={{ fontSize: 16, fontWeight: 600, marginBottom: 16, lineHeight: 1.5 }}>
                    {currentQuestion.text}
                  </h3>
                  {renderQuestion()}
                </div>
              )}
            </div>

            {/* Footer */}
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '12px 20px',
                borderTop: '1px solid var(--border)',
              }}
            >
              <button
                onClick={handlePrev}
                disabled={step === 0 && !showResult}
                style={{
                  padding: '8px 16px',
                  background: 'none',
                  border: '1px solid var(--border)',
                  borderRadius: 8,
                  color: step === 0 && !showResult ? 'var(--text-secondary)' : 'var(--text-primary)',
                  cursor: step === 0 && !showResult ? 'default' : 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 4,
                  fontSize: 14,
                }}
              >
                <ChevronLeft size={16} />
                上一步
              </button>

              {showResult ? (
                <button
                  onClick={handleSendToAI}
                  style={{
                    padding: '8px 20px',
                    background: 'var(--accent)',
                    border: 'none',
                    borderRadius: 8,
                    color: '#fff',
                    cursor: 'pointer',
                    fontSize: 14,
                    fontWeight: 600,
                  }}
                >
                  让 AI 分析
                </button>
              ) : (
                <button
                  onClick={handleNext}
                  style={{
                    padding: '8px 16px',
                    background: 'var(--accent)',
                    border: 'none',
                    borderRadius: 8,
                    color: '#fff',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 4,
                    fontSize: 14,
                  }}
                >
                  {isLastStep ? '查看结果' : '下一步'}
                  <ChevronRight size={16} />
                </button>
              )}
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
