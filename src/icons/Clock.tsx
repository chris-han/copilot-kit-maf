import { Clock as LucideClock, LucideProps } from 'lucide-react';

const Clock = ({ className, ...props }: LucideProps) => {
  return <LucideClock className={className} {...props} />;
};

export default Clock;